"""交互式 op-DAG 浏览器 → 自包含 HTML（纯 Python 生成 + 内联 JS/SVG，零外部依赖）。

按模型结构层层钻取（模型 → 层 → Cell → op-DAG）；主画布画 op 图，**悬停算子高亮它 + 上下游 + 连线，
右侧详情面板给出:op 类型/模块/源码 file:line/输入输出张量(shape·dtype)/要存的激活字节/上游算子/下游算子**。

用法：python build_opdag_explorer.py   →   analysis/viz/opdag_explorer.html
"""
import json
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from validate_dsv3 import build_dsv3_spec
from cost_eval.opdag.module_resolver import resolve_layer_spec, ResolvedSpec
from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.shape_infer import infer_shapes
from cost_eval.opdag.bprop_rules import derive_saves
from cost_eval.opdag.consumer import save_bytes

MiB = 2 ** 20
MF = r"E:\97-codes\torch_parallel\mindformers\mindformers"
_SPEC_FLAGS = {"multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
               "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
               "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
               "use_interleaved_weight_layout_mlp": True}
_MLA_FLAGS = {"use_dsa": False, "use_flash_attention": True, "use_eod_attn_mask_compression": False,
              "cp": 1, "cp_ds": 1, "input_layout": "BNSD", "q_lora_rank": 1536,
              "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32"}
_FFN_FLAGS = {"moe_token_dispatcher_type": "alltoall", "compute_dtype": "bf16", "add_bias_linear": False}


def _ref(r):
    p = (r or "").split(":")
    return {"name": p[0] if p else "", "shape": p[1] if len(p) > 1 else "?", "dtype": p[2] if len(p) > 2 else ""}


def _is_weight_shape(shape):
    """无 token/序列轴（S / cap）的张量视为**权重（持久参数）**，非激活。"""
    s = shape or ""
    return not (s.startswith("S") or "·S·" in s or "·S" == s[-2:] or "cap" in s)


def dag_to_json(dag, dims):
    """OpDAG → 前端 JSON。**消费者视角**：按 derive_saves 的 op_id（pin 该 save 的算子）把每个 save
    归到"要存它的算子"上——直接回答"当前算子为反向要存的激活多大"。三态:**saved(存激活)** /
    **param(只存权重,持久)** / **transient**。derive_saves 已按名去重,故无双算;权重不计入激活总和。"""
    by_consumer = {}
    act_total = 0
    for s in derive_saves(dag):
        try:
            b = save_bytes(s, dims) or 0
        except Exception:
            b = 0
        w = _is_weight_shape(s.sym_shape)
        by_consumer.setdefault(s.op_id, []).append(
            {"name": s.name, "bytes": b, "dtype": s.dtype, "shape": s.sym_shape, "weight": w})
        if not w:
            act_total += b
    nodes = []
    for n in dag.nodes:
        pinned = by_consumer.get(n.id, [])
        act = sum(p["bytes"] for p in pinned if not p["weight"])
        par = sum(p["bytes"] for p in pinned if p["weight"])
        role = "saved" if act > 0 else ("param" if par > 0 else "transient")
        nodes.append({
            "id": n.id, "op": n.op, "module": n.module, "src": n.src,
            "ins": [_ref(x) for x in n.ins], "out": _ref(n.out),
            "role": role, "saved": role == "saved",
            "saved_bytes": act, "param_bytes": par,
            "pinned": pinned,       # 该算子为反向要存的张量清单（激活+权重）
        })
    return {"cell": dag.cell, "nodes": nodes, "edges": [list(e) for e in dag.edges],
            "act_total_mib": round(act_total / MiB, 1)}


def build():
    spec, d, fl = build_dsv3_spec(8)
    top = resolve_layer_spec(MF, _SPEC_FLAGS)
    mla = infer_shapes(extract_cell(MF, "parallel_core/training_graph/transformer/multi_latent_attention.py",
                                    "MLASelfAttention", top.submodules["self_attention"], _MLA_FLAGS,
                                    present_params={"rotary_pos_emb"}), {"x": "S·B·H"})
    ffn = infer_shapes(extract_cell(MF, "parallel_core/training_graph/transformer/moe/ffn.py",
                                    "FFNGroupedGEMM", ResolvedSpec(cell="FFNGroupedGEMM", submodules={}), _FFN_FLAGS),
                       {"tokens": "S·B·H", "dispatched_input": "E·cap·H",
                        "w1": "(E·H)·(2·moe_ffn)", "w2": "(E·moe_ffn)·H"})
    cells = {"MLA": dag_to_json(mla, d), "MoE-FFN": dag_to_json(ffn, d)}
    # 模型结构树（层 → 该层含的 Cell）。dense 层 = MLA + dense-MLP；MoE 层 = MLA + MoE。
    structure = [
        {"name": "embedding", "cells": [], "note": "word embedding"},
        {"name": "mla_dense (L1)", "cells": ["MLA"], "note": "dense MLA 层（first_k_dense_replace=1）"},
        {"name": "mla_moe (L2–L8)", "cells": ["MLA", "MoE-FFN"], "note": "MoE MLA 层 ×7"},
        {"name": "lm_head", "cells": [], "note": "logits + CE loss（loss 区大头）"},
    ]
    data = {"cells": cells, "structure": structure,
            "dims": {"S": d.S, "B": d.B, "H": d.H, "E": d.n_experts, "moe_F": d.moe_F,
                     "q_lora_rank": d.q_lora_rank, "kv_lora_rank": d.kv_lora_rank}}
    return _PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


_PAGE = r"""<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#161b24;--mut:#6a7280;--line:#e3e6ec;--blue:#2f4b7c;--saved:#c0392b;--mono:ui-monospace,Consolas,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.top{padding:20px 26px 14px;border-bottom:1px solid var(--line);background:var(--card)}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--blue)}
h1{font-size:22px;margin:8px 0 2px;letter-spacing:-.01em}
.top p{margin:0;color:var(--mut);font-size:13px}
.layout{display:grid;grid-template-columns:230px 1fr 320px;gap:0;height:calc(100vh - 88px);min-height:560px}
.side{border-right:1px solid var(--line);background:var(--card);overflow-y:auto;padding:16px 14px}
.side h3{font:600 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:4px 0 10px}
.layer{border:1px solid var(--line);border-radius:9px;padding:9px 11px;margin-bottom:9px;background:#fafbfc}
.layer b{font-size:13px}.layer .note{color:var(--mut);font-size:11px;margin-top:2px}
.cellbtn{display:inline-block;margin:6px 6px 0 0;padding:4px 10px;border-radius:6px;border:1px solid var(--line);
  background:#fff;font:600 12px/1 var(--mono);cursor:pointer;color:var(--blue)}
.cellbtn:hover{background:#eef2f8}.cellbtn.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.canvas{position:relative;overflow:auto;background:
  linear-gradient(90deg,#eef0f3 1px,transparent 1px) 0 0/26px 26px,
  linear-gradient(#eef0f3 1px,transparent 1px) 0 0/26px 26px,var(--bg)}
.detail{border-left:1px solid var(--line);background:var(--card);overflow-y:auto;padding:18px 16px}
.detail .ph{color:var(--mut);font-size:13px}
.badge{display:inline-block;padding:2px 9px;border-radius:5px;font:700 12px/1.5 var(--mono);color:#fff}
.kv{margin:12px 0 0;font-size:12.5px}.kv .k{color:var(--mut);font:600 10.5px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-top:12px}
.kv .v{font:12.5px/1.5 var(--mono);word-break:break-all;margin-top:3px}
.savedtag{color:var(--saved);font-weight:700}.transtag{color:var(--mut)}
.tlist{list-style:none;padding:0;margin:4px 0 0}.tlist li{padding:3px 8px;margin-top:4px;border-radius:5px;background:#f2f4f7;font:12px/1.4 var(--mono);cursor:pointer}
.tlist li:hover{background:#e6ebf2}
svg .node{cursor:pointer}
svg .node rect{transition:stroke-width .1s}
.dim{opacity:.22;transition:opacity .12s}
.hl rect{stroke:#111!important;stroke-width:3.2px!important}
.edge{stroke:#c6cad2;stroke-width:1.3;fill:none}
.edge.up{stroke:#2f4b7c;stroke-width:2.4}.edge.down{stroke:#e15759;stroke-width:2.4}
.leg{display:flex;gap:16px;flex-wrap:wrap;font:11px/1 var(--mono);color:var(--mut);padding:8px 14px;border-top:1px solid var(--line);background:var(--card)}
.leg b{display:inline-flex;align-items:center;gap:6px;font-weight:400}.leg .sw{width:13px;height:11px;border-radius:3px;display:inline-block}
</style>

<div class="top">
  <div class="eyebrow">pynative-cost-evaluator · opdag explorer</div>
  <h1>DSv3 op-DAG 交互浏览器</h1>
  <p>左侧按模型结构钻取 → 选一个 Cell → 主画布看 op 图 · <b>悬停算子高亮其上下游</b>，右侧出详情（要存的激活大小 / 源码位置 / 输入输出）。</p>
</div>
<div class="layout">
  <div class="side" id="side"></div>
  <div class="canvas" id="canvas"></div>
  <div class="detail" id="detail"><p class="ph">把鼠标放到左边任一算子上 —— 这里显示它的详情：类型、源码 file:line、输入/输出张量、<b>要存的激活字节</b>、上游 / 下游算子。</p></div>
</div>
<div class="leg">
  <b><span class="sw" style="border:3px solid #c0392b"></span>saved · 反向要存激活</b>
  <b><span class="sw" style="border:2px dashed #2f4b7c"></span>param · 权重(持久)</b>
  <b><span class="sw" style="border:1px solid #9a9a9a"></span>transient · 转瞬即释/重算</b>
  <b><span class="sw" style="background:#2f4b7c"></span>MatMul</b>
  <b><span class="sw" style="background:#59a14f"></span>Norm</b>
  <b><span class="sw" style="background:#f28e2b"></span>Activation</b>
  <b><span class="sw" style="background:#e15759"></span>FlashAttn</b>
  <b><span class="sw" style="background:#9c9c9c"></span>Cast</b>
  <b><span class="sw" style="background:#d7d7d7"></span>View</b>
  <b><span style="color:#2f4b7c">▬</span> 上游</b><b><span style="color:#e15759">▬</span> 下游</b>
</div>

<script>
const DATA = __DATA__;
const OPC = {MatMul:"#4e79a7",BMM:"#4e79a7",GroupedMatMul:"#2f4b7c",Norm:"#59a14f",Softmax:"#8cd17d",
  Activation:"#f28e2b",Elementwise:"#b07aa1",Cast:"#9c9c9c",View:"#d7d7d7",Gather:"#bab0ac",
  FlashAttention:"#e15759",Dropout:"#ff9da7"};
const MiB=1048576;
let cur=null;

function levels(nodes,edges){
  const succ={},indeg={}; nodes.forEach(n=>indeg[n.id]=0);
  edges.forEach(([a,b])=>{ if(a in indeg&&b in indeg){(succ[a]=succ[a]||[]).push(b);indeg[b]++;} });
  const lv={}; nodes.forEach(n=>lv[n.id]=0);
  const ind=Object.assign({},indeg); const q=nodes.filter(n=>ind[n.id]===0).map(n=>n.id);
  while(q.length){const u=q.shift(); (succ[u]||[]).forEach(v=>{lv[v]=Math.max(lv[v],lv[u]+1); if(--ind[v]===0)q.push(v);});}
  return lv;
}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fmtBytes(b){return b?(b/MiB).toFixed(1)+" MiB":null;}

function render(cellKey){
  cur=cellKey;
  document.querySelectorAll(".cellbtn").forEach(x=>x.classList.toggle("on",x.dataset.k===cellKey));
  const dag=DATA.cells[cellKey], nodes=dag.nodes, edges=dag.edges;
  const byId={}; nodes.forEach(n=>byId[n.id]=n);
  const pred={},succ={}; nodes.forEach(n=>{pred[n.id]=[];succ[n.id]=[];});
  edges.forEach(([a,b])=>{ if(a in succ){succ[a].push(b);} if(b in pred){pred[b].push(a);} });
  const lv=levels(nodes,edges);
  const COLW=250,ROWH=82,NW=196,NH=54,MX=26,MY=26;
  const rows={}; nodes.slice().sort((a,b)=>a.id-b.id).forEach(n=>{const l=lv[n.id];rows[l]=rows[l]||0;n._x=MX+l*COLW;n._y=MY+rows[l]*ROWH;rows[l]++;});
  const maxl=Math.max(...nodes.map(n=>lv[n.id]),0), maxr=Math.max(...Object.values(rows),1);
  const W=MX*2+maxl*COLW+NW, H=MY*2+maxr*ROWH;
  let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" font-size="11">`;
  s+=`<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0L6,3L0,6Z" fill="#b6bcc6"/></marker></defs>`;
  edges.forEach(([a,b])=>{ if(!byId[a]||!byId[b])return; const p=byId[a],q=byId[b];
    s+=`<path class="edge" id="e_${a}_${b}" d="M${p._x+NW},${p._y+NH/2} C${p._x+NW+40},${p._y+NH/2} ${q._x-40},${q._y+NH/2} ${q._x},${q._y+NH/2}" marker-end="url(#ar)"/>`; });
  nodes.forEach(n=>{ const c=OPC[n.op]||"#e0e0e0"; let stroke,sw,bl,blc,dash="";
    if(n.role==="saved"){stroke="#c0392b";sw=3;bl="💾 存激活 "+fmtBytes(n.saved_bytes);blc="#ffe08a";}
    else if(n.role==="param"){stroke="#2f4b7c";sw=2;dash=' stroke-dasharray="4 3"';bl="⚙ 存权重 "+fmtBytes(n.param_bytes);blc="#cfe0f5";}
    else{stroke="#9a9a9a";sw=1;bl="↻ transient";blc="#eee";}
    s+=`<g class="node" data-id="${n.id}"><rect x="${n._x}" y="${n._y}" width="${NW}" height="${NH}" rx="7" fill="${c}" fill-opacity="0.85" stroke="${stroke}" stroke-width="${sw}"${dash}/>`;
    s+=`<text x="${n._x+8}" y="${n._y+17}" fill="#fff" font-weight="bold">#${n.id} ${esc(n.op)}${n.module?" · "+esc(n.module):""}</text>`;
    s+=`<text x="${n._x+8}" y="${n._y+32}" fill="#f2f2f2">→ ${esc(n.out.name+":"+n.out.shape)}</text>`;
    s+=`<text x="${n._x+8}" y="${n._y+46}" fill="${blc}" font-size="10.5">${esc(bl)}</text></g>`; });
  s+=`</svg>`;
  const cv=document.getElementById("canvas"); cv.innerHTML=s;

  const gs=cv.querySelectorAll(".node");
  gs.forEach(g=>{
    const id=+g.dataset.id;
    g.addEventListener("mouseenter",()=>highlight(id,byId,pred,succ,edges,cv));
    g.addEventListener("mouseleave",()=>clearHi(cv));
    g.addEventListener("click",()=>{showDetail(byId[id],pred[id],succ[id],byId);});
  });
}

function highlight(id,byId,pred,succ,edges,cv){
  const keep=new Set([id,...pred[id],...succ[id]]);
  cv.querySelectorAll(".node").forEach(g=>{const i=+g.dataset.id; g.classList.toggle("dim",!keep.has(i)); g.classList.toggle("hl",i===id);});
  cv.querySelectorAll(".edge").forEach(e=>{e.classList.remove("up","down"); e.classList.add("dim");});
  edges.forEach(([a,b])=>{ const e=document.getElementById(`e_${a}_${b}`); if(!e)return;
    if(b===id){e.classList.remove("dim");e.classList.add("up");} if(a===id){e.classList.remove("dim");e.classList.add("down");} });
  showDetail(byId[id],pred[id],succ[id],byId);
}
function clearHi(cv){cv.querySelectorAll(".dim,.hl").forEach(x=>x.classList.remove("dim","hl")); cv.querySelectorAll(".edge").forEach(e=>e.classList.remove("up","down","dim"));}

function tensorLine(t){return `${esc(t.name)} : <b>${esc(t.shape)}</b> · ${esc(t.dtype)}`;}
function showDetail(n,pred,succ,byId){
  const c=OPC[n.op]||"#888";
  let savedLine;
  if(n.role==="saved") savedLine=`<span class="savedtag">💾 存激活 ${fmtBytes(n.saved_bytes)}</span> — 该算子为反向要存进内存的激活（fwd→bwd 驻留，占峰值）`;
  else if(n.role==="param") savedLine=`<span style="color:#2f4b7c;font-weight:700">⚙ 存权重 ${fmtBytes(n.param_bytes)}</span> — 只 pin 持久权重（不计入激活峰值）`;
  else savedLine=`<span class="transtag">↻ transient</span> — 该算子反向不 pin 激活（转瞬即释 / 重算）`;
  const pinnedHtml=(n.pinned&&n.pinned.length)
    ? n.pinned.map(p=>`<div style="padding:2px 0"><span style="color:${p.weight?'#2f4b7c':'#c0392b'};font-weight:700">${p.weight?'⚙':'💾'} ${fmtBytes(p.bytes)||'?'}</span> ${esc(p.name)} : <b>${esc(p.shape)}</b> · ${esc(p.dtype)}${p.weight?' <span style="color:#999">(权重·持久)</span>':''}</div>`).join("")
    : '<span style="color:#999">（无 · 反向不 pin）</span>';
  const upl=pred.length?pred.map(i=>`<li data-goto="${i}">↑ #${i} ${esc(byId[i].op)}${byId[i].module?" · "+esc(byId[i].module):""}</li>`).join(""):'<li style="cursor:default;color:#999">（无 · 输入边界）</li>';
  const dnl=succ.length?succ.map(i=>`<li data-goto="${i}">↓ #${i} ${esc(byId[i].op)}${byId[i].module?" · "+esc(byId[i].module):""}</li>`).join(""):'<li style="cursor:default;color:#999">（无 · 输出边界）</li>';
  document.getElementById("detail").innerHTML=`
    <span class="badge" style="background:${c}">${esc(n.op)}</span> ${n.module?`<span style="font:600 12px var(--mono);color:var(--blue)">${esc(n.module)}</span>`:""}
    <div class="kv">
      <div class="k">要存的激活</div><div class="v">${savedLine}</div>
      <div class="k">为反向存的张量清单</div><div class="v">${pinnedHtml}</div>
      <div class="k">输出张量</div><div class="v">${tensorLine(n.out)}</div>
      <div class="k">输入张量（操作数）</div><div class="v">${n.ins.map(tensorLine).join("<br>")||"（无）"}</div>
      <div class="k">源码位置</div><div class="v">${esc(n.src)}</div>
      <div class="k">上游算子（${pred.length}）</div><ul class="tlist">${upl}</ul>
      <div class="k">下游算子（${succ.length}）</div><ul class="tlist">${dnl}</ul>
    </div>`;
  document.querySelectorAll("#detail li[data-goto]").forEach(li=>li.addEventListener("click",()=>{
    const g=document.querySelector(`.node[data-id="${li.dataset.goto}"]`); if(g)g.dispatchEvent(new Event("mouseenter"));}));
}

// 侧栏结构树
(function(){
  let h=`<h3>模型结构（层层递进）</h3>`;
  DATA.structure.forEach(L=>{
    h+=`<div class="layer"><b>${esc(L.name)}</b><div class="note">${esc(L.note)}</div>`;
    L.cells.forEach(ck=>{ h+=`<button class="cellbtn" data-k="${ck}">${ck} · 激活 ${DATA.cells[ck].act_total_mib} MiB</button>`; });
    h+=`</div>`;
  });
  h+=`<h3 style="margin-top:16px">配置</h3><div class="layer" style="font:11.5px/1.6 var(--mono);color:var(--mut)">DSv3 8L · S=${DATA.dims.S} B=${DATA.dims.B}<br>H=${DATA.dims.H} E=${DATA.dims.E}<br>q_lora=${DATA.dims.q_lora_rank} kv_lora=${DATA.dims.kv_lora_rank}</div>`;
  document.getElementById("side").innerHTML=h;
  document.querySelectorAll(".cellbtn").forEach(b=>b.addEventListener("click",()=>render(b.dataset.k)));
  render("MLA");
})();
</script>"""


if __name__ == "__main__":
    html = build()
    os.makedirs("analysis/viz", exist_ok=True)
    out = "analysis/viz/opdag_explorer.html"
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, os.path.getsize(out), "bytes")
