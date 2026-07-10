"""全模型竖向 op-DAG 浏览器 + 可点击内存 timeline → 自包含 HTML（纯 Python + 内联 JS/SVG）。

竖向全模型图（对齐 mindformers 代码层次）：
  input_ids → embedding → TransformerLayer[ input_layernorm → self_attention(MLA) → +residual
  → pre_mlp_layernorm → mlp(MoE) → +residual ] ×N → final_layernorm → lm_head → loss
每算子标"要存的激活"（消费者视角）；router/dispatch/combine/shared 等**不透明段**用灰占位框诚实标出。
悬停算子高亮上下游 + 出详情；点 timeline 事件 → 该时刻各桶大小。

用法：python build_model_explorer.py  →  analysis/viz/model_explorer.html
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
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

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
    s = shape or ""
    return not (s.startswith("S") or "·S·" in s or (len(s) >= 2 and s[-2:] == "·S") or "cap" in s)


def cell_nodes(dag, dims, group):
    """一个 Cell DAG → 节点列表（消费者视角 pinned）+ 局部边。返回 (nodes, edges, first_ids, last_ids)。"""
    by_consumer = {}
    for s in derive_saves(dag):
        try:
            b = save_bytes(s, dims) or 0
        except Exception:
            b = 0
        by_consumer.setdefault(s.op_id, []).append(
            {"name": s.name, "bytes": b, "dtype": s.dtype, "shape": s.sym_shape, "weight": _is_weight_shape(s.sym_shape)})
    nodes = []
    for n in dag.nodes:
        pinned = by_consumer.get(n.id, [])
        act = sum(p["bytes"] for p in pinned if not p["weight"])
        par = sum(p["bytes"] for p in pinned if p["weight"])
        role = "saved" if act > 0 else ("param" if par > 0 else "transient")
        nodes.append({"id": n.id, "op": n.op, "module": n.module, "src": n.src, "group": group,
                      "ins": [_ref(x) for x in n.ins], "out": _ref(n.out),
                      "role": role, "saved_bytes": act, "param_bytes": par, "pinned": pinned})
    ids = {n.id for n in dag.nodes}
    dst = {b for a, b in dag.edges if a in ids}
    src = {a for a, b in dag.edges if b in ids}
    first = [n.id for n in dag.nodes if n.id not in dst]     # 无入边 = cell 入口
    last = [n.id for n in dag.nodes if n.id not in src]      # 无出边 = cell 出口
    return nodes, [list(e) for e in dag.edges], first, last


# 合成 spine / 不透明占位节点。
def _syn(op, out, group, module="", note="", role="transient", act=0):
    return {"op": op, "module": module, "src": note, "group": group, "ins": [], "out": _ref(out),
            "role": role, "saved_bytes": act, "param_bytes": 0,
            "pinned": ([{"name": _ref(out)["name"], "bytes": act, "dtype": _ref(out)["dtype"],
                         "shape": _ref(out)["shape"], "weight": False}] if act else []), "opaque": op == "Opaque"}


def build_model_graph(dims):
    mla = infer_shapes(extract_cell(MF, "parallel_core/training_graph/transformer/multi_latent_attention.py",
                                    "MLASelfAttention", resolve_layer_spec(MF, _SPEC_FLAGS).submodules["self_attention"],
                                    _MLA_FLAGS, present_params={"rotary_pos_emb"}), {"x": "S·B·H"})
    ffn = infer_shapes(extract_cell(MF, "parallel_core/training_graph/transformer/moe/ffn.py",
                                    "FFNGroupedGEMM", ResolvedSpec(cell="FFNGroupedGEMM", submodules={}), _FFN_FLAGS),
                       {"tokens": "S·B·H", "dispatched_input": "E·cap·H", "w1": "(E·H)·(2·moe_ffn)", "w2": "(E·moe_ffn)·H"})
    mla_n, mla_e, mla_first, mla_last = cell_nodes(mla, dims, "self_attention · MLASelfAttention")
    ffn_n, ffn_e, ffn_first, ffn_last = cell_nodes(ffn, dims, "mlp · FFNGroupedGEMM(experts)")

    # 竖向 spine 段（每段一个/一组节点）。opaque=不透明段（router/dispatch/combine/shared，静态不可分解）。
    spine = [
        ("in", _syn("Input", "input_ids:B·S:int32", "input", note="模型入口")),
        ("emb", _syn("Embedding", "hidden:S·B·H:bf16", "embedding · Embedding", module="VocabEmbedding",
                     note="parallel_core/.../language_model_embedding.py", role="saved", act=int(dims.S * dims.B * dims.H * 2))),
        ("ln1", _syn("Norm", "ln1:S·B·H:bf16", "input_layernorm · RMSNorm", module="RMSNorm",
                     note="transformer_layer.py:208", role="saved", act=int(dims.S * dims.B * dims.H * 4))),
        ("MLA", None),   # 占位:后面塞 mla_n
        ("bda1", _syn("Elementwise", "h1:S·B·H:bf16", "self_attn_bda · residual add", module="Add",
                      note="transformer_layer.py 残差")),
        ("ln2", _syn("Norm", "ln2:S·B·H:bf16", "pre_mlp_layernorm · RMSNorm", module="RMSNorm",
                     note="transformer_layer.py:~300", role="saved", act=int(dims.S * dims.B * dims.H * 4))),
        ("route", _syn("Opaque", "dispatched:E·cap·H:bf16", "mlp · TopKRouter+dispatch(不透明)", module="AllToAll",
                       note="router topk/argmax + permute/capacity-pad —— 数据依赖,静态不可分解")),
        ("MoE", None),   # 占位:塞 ffn_n
        ("combine", _syn("Opaque", "moe_out:S·B·H:bf16", "mlp · combine(不透明)", module="AllToAll",
                         note="unpermute + all-to-all 还原")),
        ("shared", _syn("Opaque", "shared_out:S·B·H:bf16", "mlp · shared_expert(未提取)", module="SharedExpert",
                        note="SharedExpertMLPInterleaved（super().construct 惯用法未提取）")),
        ("bda2", _syn("Elementwise", "h2:S·B·H:bf16", "mlp_bda · residual add", module="Add",
                      note="transformer_layer.py 残差")),
        ("fnorm", _syn("Norm", "hfinal:S·B·H:bf16", "final_layernorm · RMSNorm", module="RMSNorm",
                       note="transformer_block 末", role="saved", act=int(dims.S * dims.B * dims.H * 4))),
        ("head", _syn("MatMul", "logits:S·B·vocab:bf16", "lm_head · ColumnParallelLinear", module="ColumnParallelLinear",
                      note="head.py", role="saved", act=int(dims.S * dims.B * dims.vocab * 2))),
        ("loss", _syn("Softmax", "loss:1:fp32", "loss · CrossEntropy(logsoftmax+NLL)", module="CE",
                      note="head.py:loss —— 满 vocab fp32 中间量(loss 区大头)", role="saved",
                      act=int(dims.S * dims.B * dims.vocab * 4))),
    ]

    nodes, edges = [], []
    gid = [0]
    def add(nd):
        nd = dict(nd); nd["gid"] = gid[0]; gid[0] += 1; nodes.append(nd); return nd["gid"]

    remap_prev_last = None  # 上一段的出口 gid 列表（连到本段入口）
    layer_note_gid = None
    for key, nd in spine:
        if key == "MLA":
            local = {n["id"]: add(n) for n in mla_n}
            for a, b in mla_e:
                if a in local and b in local: edges.append([local[a], local[b]])
            if remap_prev_last:
                for g in remap_prev_last:
                    for fid in mla_first: edges.append([g, local[fid]])
            remap_prev_last = [local[i] for i in mla_last]
        elif key == "MoE":
            local = {n["id"]: add(n) for n in ffn_n}
            for a, b in ffn_e:
                if a in local and b in local: edges.append([local[a], local[b]])
            if remap_prev_last:
                for g in remap_prev_last:
                    for fid in ffn_first: edges.append([g, local[fid]])
            remap_prev_last = [local[i] for i in ffn_last]
        else:
            g = add(nd)
            if remap_prev_last:
                for pg in remap_prev_last: edges.append([pg, g])
            remap_prev_last = [g]
            if key == "bda2": layer_note_gid = g
    return {"nodes": nodes, "edges": edges, "layer_end_gid": layer_note_gid}


def build_timeline(spec, dims):
    d = dims; d.B = 1
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    rep = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                    HardwareSpec(max_device_memory=64 * 2 ** 30, framework_reserve=0),
                    RecomputeSpec("None"), SwapSpec()).evaluate(record_timeline=True)
    sp = rep.per_stage[rep.tightest_stage]
    BK = ["persistent", "act_live", "kept_frag", "gather_buf", "grad_buf", "recomp_scratch",
          "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "optstep", "framework"]
    events = [{"event": s.event, "total": round(s.total_bytes / MiB, 1),
               "buckets": {k: round(getattr(s.breakdown, k, 0) / MiB, 1) for k in BK if getattr(s.breakdown, k, 0)}}
              for s in sp.timeline]
    return {"events": events, "peak": round(sp.peak_bytes / MiB, 1), "peak_event": sp.peak_event}


def build():
    spec, d, fl = build_dsv3_spec(8)
    data = {"graph": build_model_graph(d), "timeline": build_timeline(spec, d),
            "dims": {"S": d.S, "B": d.B, "H": d.H, "E": d.n_experts, "vocab": d.vocab,
                     "q_lora_rank": d.q_lora_rank, "kv_lora_rank": d.kv_lora_rank, "N": d.n_layers}}
    return _PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


_PAGE = r"""__CSS__
<div class="top">
  <div class="eyebrow">pynative-cost-evaluator · model explorer</div>
  <h1>DSv3 全模型内存图（竖向 · 对齐 mindformers 层次）</h1>
  <p>input_ids → embedding → TransformerLayer(input_layernorm→MLA→残差→pre_mlp_layernorm→MoE→残差)×N → final_layernorm → lm_head → loss。<b>悬停算子</b>看要存的激活 + 上下游；<b>点右侧 timeline</b> 看该时刻各桶开销。灰色 = 不透明段。</p>
</div>
<div class="layout">
  <div class="graph" id="graph"></div>
  <div class="right">
    <div class="detail" id="detail"><p class="ph">悬停左侧任一算子 → 这里显示：类型 / 要存的激活字节 / 为反向存的张量清单 / 源码 / 上游·下游算子。</p></div>
    <div class="tl" id="tl"></div>
  </div>
</div>
__SCRIPT__"""


_CSS = r"""<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#161b24;--mut:#6a7280;--line:#e3e6ec;--blue:#2f4b7c;--saved:#c0392b;--mono:ui-monospace,Consolas,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.top{padding:18px 24px 12px;border-bottom:1px solid var(--line);background:var(--card)}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--blue)}
h1{font-size:21px;margin:7px 0 2px}.top p{margin:0;color:var(--mut);font-size:12.5px;max-width:118ch}
.layout{display:grid;grid-template-columns:1fr 340px;height:calc(100vh - 82px);min-height:560px}
.graph{overflow:auto;background:linear-gradient(#eef0f3 1px,transparent 1px) 0 0/24px 24px,var(--bg)}
.right{border-left:1px solid var(--line);background:var(--card);display:flex;flex-direction:column;overflow:hidden}
.detail{flex:1;overflow-y:auto;padding:16px 15px;border-bottom:1px solid var(--line)}
.detail .ph{color:var(--mut);font-size:12.5px}
.tl{height:320px;overflow:auto;padding:10px 12px}
.badge{display:inline-block;padding:2px 9px;border-radius:5px;font:700 12px/1.5 var(--mono);color:#fff}
.kv .k{color:var(--mut);font:600 10px/1 var(--mono);letter-spacing:.07em;text-transform:uppercase;margin-top:12px}
.kv .v{font:12px/1.5 var(--mono);word-break:break-all;margin-top:3px}
.savedtag{color:var(--saved);font-weight:700}.transtag{color:var(--mut)}
.tlist{list-style:none;padding:0;margin:4px 0 0}.tlist li{padding:3px 8px;margin-top:4px;border-radius:5px;background:#f2f4f7;font:11.5px/1.4 var(--mono);cursor:pointer}
.tlist li:hover{background:#e6ebf2}
svg .node{cursor:pointer}.dim{opacity:.18}.hl rect{stroke:#111!important;stroke-width:3.4px!important}
.edge{stroke:#c6cad2;stroke-width:1.3;fill:none}.edge.up{stroke:#2f4b7c;stroke-width:2.6}.edge.down{stroke:#e15759;stroke-width:2.6}
.grp{fill:#fff;fill-opacity:.55;stroke:#cfd6e0;stroke-width:1;stroke-dasharray:5 4}
.grplab{font:600 11px var(--mono);fill:#2f4b7c}
.barrow{display:flex;align-items:center;gap:8px;padding:2px 0;font:11.5px/1.3 var(--mono)}
.barrow .bl{width:96px;text-align:right;color:#333;flex-shrink:0}
.bartrack{flex:1;height:16px;background:#eef0f3;border-radius:4px;overflow:hidden}.barfill{height:100%}
.barrow .bv{width:64px;text-align:right;color:#555;flex-shrink:0}
.tlhint{font:11px var(--mono);color:var(--mut);margin:0 0 8px}
</style>"""


_SCRIPT = r"""<script>
const DATA=__DATA__;
const OPC={MatMul:"#4e79a7",BMM:"#4e79a7",GroupedMatMul:"#2f4b7c",Norm:"#59a14f",Softmax:"#e15759",Activation:"#f28e2b",Elementwise:"#b07aa1",Cast:"#9c9c9c",View:"#d7d7d7",Gather:"#bab0ac",FlashAttention:"#e15759",Dropout:"#ff9da7",Input:"#556",Embedding:"#7c6",Opaque:"#b9bec7"};
const BKC={persistent:"#6b6b6b",act_live:"#4e79a7",kept_frag:"#c0392b",gather_buf:"#59a14f",grad_buf:"#f28e2b",recomp_scratch:"#b07aa1",bwd_scratch:"#e15759",bwd_working_set:"#8cd17d",swap_buf:"#76b7b2",workspace:"#bab0ac",optstep:"#ff9da7",framework:"#d7d7d7"};
const MiB=1048576;
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fb(b){return b?(b/MiB).toFixed(1)+" MiB":"0";}
function fmib(m){return (m>=1024?(m/1024).toFixed(1)+" GiB":m.toFixed(0)+" MiB");}

// ── 竖向布局：topo level → y，同 level 并排 → x ──
function layout(nodes,edges){
  const succ={},indeg={}; nodes.forEach(n=>indeg[n.gid]=0);
  edges.forEach(([a,b])=>{(succ[a]=succ[a]||[]).push(b); if(b in indeg)indeg[b]++;});
  const lv={}; nodes.forEach(n=>lv[n.gid]=0);
  const ind=Object.assign({},indeg); let q=nodes.filter(n=>ind[n.gid]===0).map(n=>n.gid);
  while(q.length){const u=q.shift();(succ[u]||[]).forEach(v=>{lv[v]=Math.max(lv[v],lv[u]+1); if(--ind[v]===0)q.push(v);});}
  const rows={}; nodes.slice().sort((a,b)=>a.gid-b.gid).forEach(n=>{const l=lv[n.gid];rows[l]=rows[l]||0;n._lane=rows[l]++;n._lv=l;});
  return lv;
}
function render(){
  const g=DATA.graph, nodes=g.nodes, edges=g.edges;
  const byId={}; nodes.forEach(n=>byId[n.gid]=n);
  const pred={},succ={}; nodes.forEach(n=>{pred[n.gid]=[];succ[n.gid]=[];});
  edges.forEach(([a,b])=>{if(a in succ)succ[a].push(b); if(b in pred)pred[b].push(a);});
  layout(nodes,edges);
  const NW=232,NH=50,COLW=250,ROWH=76,MX=150,MY=30;
  nodes.forEach(n=>{n._x=MX+n._lane*COLW; n._y=MY+n._lv*ROWH;});
  const maxlane=Math.max(...nodes.map(n=>n._lane),0), maxlv=Math.max(...nodes.map(n=>n._lv),0);
  const W=MX+ (maxlane+1)*COLW+40, H=MY+(maxlv+1)*ROWH+30;
  // 分组框（按 group，取该组节点包围盒）
  const grp={}; nodes.forEach(n=>{(grp[n.group]=grp[n.group]||[]).push(n);});
  let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" font-size="11">`;
  s+=`<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0L6,3L0,6Z" fill="#b6bcc6"/></marker></defs>`;
  Object.entries(grp).forEach(([gname,ns])=>{
    if(ns.length<2 && !gname.includes("MLA") && !gname.includes("FFN"))return;
    const x0=Math.min(...ns.map(n=>n._x))-10,y0=Math.min(...ns.map(n=>n._y))-16,
          x1=Math.max(...ns.map(n=>n._x+NW))+10,y1=Math.max(...ns.map(n=>n._y+NH))+8;
    s+=`<rect class="grp" x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" rx="10"/><text class="grplab" x="${x0+8}" y="${y0-4}">${esc(gname)}</text>`;
  });
  edges.forEach(([a,b])=>{const p=byId[a],q=byId[b]; if(!p||!q)return;
    s+=`<path class="edge" id="e_${a}_${b}" d="M${p._x+NW/2},${p._y+NH} C${p._x+NW/2},${p._y+NH+28} ${q._x+NW/2},${q._y-28} ${q._x+NW/2},${q._y}" marker-end="url(#ar)"/>`;});
  nodes.forEach(n=>{const c=OPC[n.op]||"#e0e0e0"; let stroke,sw,bl,blc,dash="";
    if(n.opaque){stroke="#9aa0aa";sw=1.5;dash=' stroke-dasharray="6 4"';bl="⊘ 不透明段";blc="#eee";}
    else if(n.role==="saved"){stroke="#c0392b";sw=3;bl="💾 存激活 "+fb(n.saved_bytes);blc="#ffe08a";}
    else if(n.role==="param"){stroke="#2f4b7c";sw=2;dash=' stroke-dasharray="4 3"';bl="⚙ 存权重 "+fb(n.param_bytes);blc="#cfe0f5";}
    else{stroke="#9a9a9a";sw=1;bl="↻ transient";blc="#eee";}
    s+=`<g class="node" data-id="${n.gid}"><rect x="${n._x}" y="${n._y}" width="${NW}" height="${NH}" rx="7" fill="${c}" fill-opacity="0.86" stroke="${stroke}" stroke-width="${sw}"${dash}/>`;
    s+=`<text x="${n._x+9}" y="${n._y+18}" fill="#fff" font-weight="bold">${esc(n.op)}${n.module?" · "+esc(n.module):""}</text>`;
    s+=`<text x="${n._x+9}" y="${n._y+33}" fill="#f2f2f2">→ ${esc(n.out.name+":"+n.out.shape)}</text>`;
    s+=`<text x="${n._x+9}" y="${n._y+45}" fill="${blc}" font-size="10">${esc(bl)}</text></g>`;});
  s+=`</svg>`;
  const cv=document.getElementById("graph"); cv.innerHTML=s;
  cv.querySelectorAll(".node").forEach(gg=>{const id=+gg.dataset.id;
    gg.addEventListener("mouseenter",()=>hi(id,byId,pred,succ,edges,cv));
    gg.addEventListener("mouseleave",()=>clr(cv));});
}
function hi(id,byId,pred,succ,edges,cv){
  const keep=new Set([id,...pred[id],...succ[id]]);
  cv.querySelectorAll(".node").forEach(g=>{const i=+g.dataset.id;g.classList.toggle("dim",!keep.has(i));g.classList.toggle("hl",i===id);});
  cv.querySelectorAll(".edge").forEach(e=>{e.classList.remove("up","down");e.classList.add("dim");});
  edges.forEach(([a,b])=>{const e=document.getElementById(`e_${a}_${b}`);if(!e)return;
    if(b===id){e.classList.remove("dim");e.classList.add("up");}if(a===id){e.classList.remove("dim");e.classList.add("down");}});
  detail(byId[id],pred[id],succ[id],byId);
}
function clr(cv){cv.querySelectorAll(".dim,.hl").forEach(x=>x.classList.remove("dim","hl"));cv.querySelectorAll(".edge").forEach(e=>e.classList.remove("up","down","dim"));}
function tl(t){return `${esc(t.name)} : <b>${esc(t.shape)}</b> · ${esc(t.dtype)}`;}
function detail(n,pred,succ,byId){
  const c=OPC[n.op]||"#888";
  let line;
  if(n.opaque) line=`<span style="color:#7a828e;font-weight:700">⊘ 不透明段</span> — ${esc(n.src)}（静态不可分解，退标定 margin）`;
  else if(n.role==="saved") line=`<span class="savedtag">💾 存激活 ${fb(n.saved_bytes)}</span> — 该算子为反向要存进内存的激活`;
  else if(n.role==="param") line=`<span style="color:#2f4b7c;font-weight:700">⚙ 存权重 ${fb(n.param_bytes)}</span> — 持久权重（不占激活峰）`;
  else line=`<span class="transtag">↻ transient</span> — 反向不 pin 激活`;
  const pin=(n.pinned&&n.pinned.length)?n.pinned.map(p=>`<div><span style="color:${p.weight?'#2f4b7c':'#c0392b'};font-weight:700">${p.weight?'⚙':'💾'} ${fb(p.bytes)}</span> ${esc(p.name)}:<b>${esc(p.shape)}</b>·${esc(p.dtype)}</div>`).join(""):'<span style="color:#999">（无）</span>';
  const U=pred.length?pred.map(i=>`<li data-goto="${i}">↑ ${esc(byId[i].op)}${byId[i].module?" · "+esc(byId[i].module):""}</li>`).join(""):'<li style="color:#999;cursor:default">（模型入口）</li>';
  const D=succ.length?succ.map(i=>`<li data-goto="${i}">↓ ${esc(byId[i].op)}${byId[i].module?" · "+esc(byId[i].module):""}</li>`).join(""):'<li style="color:#999;cursor:default">（模型出口）</li>';
  document.getElementById("detail").innerHTML=`<span class="badge" style="background:${c}">${esc(n.op)}</span> <span style="font:600 12px var(--mono);color:var(--blue)">${esc(n.module||"")}</span>
    <div class="kv"><div class="k">要存的激活</div><div class="v">${line}</div>
    <div class="k">为反向存的张量</div><div class="v">${pin}</div>
    <div class="k">输出</div><div class="v">${tl(n.out)}</div>
    <div class="k">代码层次 / 源码</div><div class="v">${esc(n.group)}<br>${esc(n.src||"")}</div>
    <div class="k">上游</div><ul class="tlist">${U}</ul><div class="k">下游</div><ul class="tlist">${D}</ul></div>`;
  document.querySelectorAll("#detail li[data-goto]").forEach(li=>li.addEventListener("click",()=>{const g=document.querySelector(`.node[data-id="${li.dataset.goto}"]`);if(g)g.dispatchEvent(new Event("mouseenter"));}));
}
// ── 内存 timeline（点事件 → 各桶）──
function renderTL(){
  const ev=DATA.timeline.events, peak=DATA.timeline.peak;
  const W=316, PT=8, ROWH=17, plotw=W-70;
  const ymax=peak*1.05;
  let s=`<p class="tlhint">内存时间线（FWD→BWD，${ev.length} 事件）· 峰值 ${fmib(peak)} @ ${esc(DATA.timeline.peak_event)}。<b>点某事件</b>看该刻各桶 ↓</p>`;
  s+=`<svg width="${W}" height="${PT*2+ev.length*ROWH}" xmlns="http://www.w3.org/2000/svg" font:10px var(--mono)>`;
  ev.forEach((e,i)=>{const y=PT+i*ROWH; let x=64;
    s+=`<text x="60" y="${y+11}" text-anchor="end" font-size="9" fill="#888">${esc(e.event)}</text>`;
    s+=`<rect class="tlbar" data-i="${i}" x="64" y="${y+2}" width="${plotw}" height="${ROWH-4}" fill="transparent" style="cursor:pointer"/>`;
    Object.entries(e.buckets).forEach(([k,v])=>{const w=v/ymax*plotw; s+=`<rect x="${x}" y="${y+2}" width="${w.toFixed(1)}" height="${ROWH-4}" fill="${BKC[k]||'#ccc'}" fill-opacity=".85"/>`; x+=w;});
    if(i===ev.findIndex(z=>z.total===peak)) s+=`<text x="${64+plotw+2}" y="${y+11}" font-size="9" fill="#c0392b">★</text>`;
  });
  s+=`</svg>`;
  const el=document.getElementById("tl"); el.innerHTML=s;
  el.querySelectorAll(".tlbar").forEach(b=>b.addEventListener("click",()=>showEvent(+b.dataset.i)));
  showEvent(ev.findIndex(z=>z.total===peak));  // 默认显示峰值事件
}
function showEvent(i){
  const e=DATA.timeline.events[i];
  const bs=Object.entries(e.buckets).sort((a,b)=>b[1]-a[1]);
  const maxv=Math.max(...bs.map(x=>x[1]),1);
  let h=`<div style="font:600 12px var(--mono);margin-bottom:6px">事件 <b>${esc(e.event)}</b> · 总占用 ${fmib(e.total)}</div>`;
  bs.forEach(([k,v])=>{h+=`<div class="barrow"><span class="bl">${esc(k)}</span><span class="bartrack"><span class="barfill" style="width:${v/maxv*100}%;background:${BKC[k]||'#ccc'}"></span></span><span class="bv">${v.toFixed(0)} · ${(v/e.total*100).toFixed(0)}%</span></div>`;});
  document.getElementById("detail").innerHTML=`<div style="font:600 11px var(--mono);color:#2f4b7c;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Timeline 事件 · 各桶开销</div>`+h+`<p style="color:#999;font-size:11.5px;margin-top:12px">悬停左图任一算子可切回算子详情。</p>`;
}
render(); renderTL();
</script>"""

_PAGE = _PAGE.replace("__CSS__", _CSS).replace("__SCRIPT__", _SCRIPT)


if __name__ == "__main__":
    html = build()
    os.makedirs("analysis/viz", exist_ok=True)
    out = "analysis/viz/model_explorer.html"
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, os.path.getsize(out), "bytes")
