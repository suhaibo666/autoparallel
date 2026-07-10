"""交互式内存实验台（本地服务，stdlib 零依赖）：浏览器改并行切分 → 实时重算 → 刷新内存开销；
PP 每个 stage 单独展示。左侧静态全模型结构图，右侧配置面板 + per-stage 标签 + timeline + 各桶。

启动：python serve_explorer.py   →   打开 http://127.0.0.1:8765
（模型在 Python 侧,故用本地服务实时重算,非自包含 artifact——artifact 只能静态。）
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.stdout.reconfigure(encoding="utf-8")
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB = 2 ** 20
BK = ["persistent", "act_live", "kept_frag", "gather_buf", "grad_buf", "recomp_scratch",
      "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "optstep", "framework"]
_ATTN = lambda N: {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, N + 1)}
_MLP = lambda N: {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"} for lid in range(1, N + 1)}


def eval_config(p):
    """p: dict(dp,tp,ep,pp,cp,method,recompute,select,layers,batch)。→ per-stage 内存 JSON。"""
    N = int(p.get("layers", 8)); B = int(p.get("batch", 1))
    dp = int(p.get("dp", 2)); tp = int(p.get("tp", 1)); ep = int(p.get("ep", 1))
    pp = int(p.get("pp", 1)); cp = int(p.get("cp", 1)); method = p.get("method", "colossal")
    rmode = p.get("recompute", "None"); sel = p.get("select", "attn")
    spec, d, fl = build_dsv3_spec(N); d.B = B
    if rmode == "full":
        rc = RecomputeSpec("full", full_layers=fl)
    elif rmode == "select":
        ops = _ATTN(N) if sel == "attn" else (_MLP(N) if sel == "mlp" else
              {lid: _ATTN(N)[lid] | _MLP(N)[lid] for lid in range(1, N + 1)})
        rc = RecomputeSpec("select", select_ops=ops)
    else:
        rc = RecomputeSpec("None")
    mbs = pp if pp > 1 else 1
    pc = ParallelConfig(dp_shard=dp, tp=tp, ep=ep, pp=pp, cp=cp, sequence_parallel=(dp > 1 or tp > 1),
                        num_microbatches=mbs, context_parallel_method=method)
    rep = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                    HardwareSpec(max_device_memory=64 * 2 ** 30, framework_reserve=0), rc, SwapSpec()
                    ).evaluate(record_timeline=True)
    stages = []
    L_per = max(1, N // pp)
    for sp in rep.per_stage:
        # 该 stage 的 transformer 层范围（近似：连续切分；stage0 含 embedding、末 stage 含 head）。
        lo = sp.stage * L_per + 1; hi = min(N, (sp.stage + 1) * L_per)
        rng = f"L{lo}–L{hi}" + (" + embedding" if sp.stage == 0 else "") + (" + head/loss" if sp.stage == pp - 1 else "")
        stages.append({
            "stage": sp.stage, "peak": round(sp.peak_bytes / MiB, 1), "peak_event": sp.peak_event,
            "oom": sp.oom, "layers": rng,
            "buckets_at_peak": {k: round(getattr(sp.breakdown, k, 0) / MiB, 1) for k in BK if getattr(sp.breakdown, k, 0)},
            "timeline": [{"event": s.event, "total": round(s.total_bytes / MiB, 1),
                          "buckets": {k: round(getattr(s.breakdown, k, 0) / MiB, 1) for k in BK if getattr(s.breakdown, k, 0)}}
                         for s in sp.timeline],
        })
    return {"ok": True, "config": p, "tightest": rep.tightest_stage,
            "device_peak": round(max(s["peak"] for s in stages), 1), "stages": stages}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(PAGE, "text/html"); return
        if u.path == "/api/eval":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                self._send(json.dumps(eval_config(q), ensure_ascii=False))
            except Exception as e:
                self._send(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
            return
        self.send_response(404); self.end_headers()


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>DSv3 内存实验台</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#161b24;--mut:#6a7280;--line:#e3e6ec;--blue:#2f4b7c;--saved:#c0392b;--mono:ui-monospace,Consolas,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.top{padding:16px 22px 10px;border-bottom:1px solid var(--line);background:var(--card)}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--blue)}
h1{font-size:20px;margin:6px 0 2px}.top p{margin:0;color:var(--mut);font-size:12.5px}
.cfg{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-top:12px}
.fld{display:flex;flex-direction:column;gap:3px}.fld label{font:600 10px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.fld select,.fld input{font:13px var(--mono);padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;min-width:70px}
.kpis{display:flex;gap:12px;margin-left:auto}
.kpi{background:#fafbfc;border:1px solid var(--line);border-radius:9px;padding:8px 14px;text-align:right}
.kpi .n{font:700 20px/1 var(--mono)}.kpi .n.big{color:var(--saved)}.kpi .n.oom{color:#fff;background:#c0392b;padding:2px 6px;border-radius:5px}
.kpi .t{color:var(--mut);font-size:11px;margin-top:3px}
.wrap{padding:16px 22px 40px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.tab{padding:6px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font:600 12px var(--mono)}
.tab.on{background:var(--blue);color:#fff;border-color:var(--blue)}.tab.oom{border-color:#c0392b;color:#c0392b}.tab.on.oom{background:#c0392b;color:#fff}
.grid{display:grid;grid-template-columns:1fr 360px;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card header{padding:12px 16px;border-bottom:1px solid var(--line);font-weight:600;font-size:14px}
.card .b{padding:12px 16px}
.tlrow{display:flex;align-items:center;gap:8px;font:11px var(--mono);padding:1px 0;cursor:pointer;border-radius:4px}
.tlrow:hover{background:#f0f2f5}.tlrow.on{background:#eaf0f8}
.tlrow .e{width:70px;text-align:right;color:#888;flex-shrink:0}
.tlbar{flex:1;height:15px;background:#eef0f3;border-radius:3px;overflow:hidden;display:flex}
.tlrow .tv{width:60px;text-align:right;color:#555;flex-shrink:0}
.barrow{display:flex;align-items:center;gap:8px;padding:2.5px 0;font:12px var(--mono)}
.barrow .bl{width:118px;text-align:right;color:#333;flex-shrink:0}
.bartrack{flex:1;height:17px;background:#eef0f3;border-radius:4px;overflow:hidden}.barfill{height:100%}
.barrow .bv{width:78px;text-align:right;color:#555;flex-shrink:0;font-variant-numeric:tabular-nums}
.err{color:#c0392b;font:13px var(--mono);padding:10px 0}
.desc{color:var(--mut);font-size:12px;margin:0 0 10px}
.busy{opacity:.5}
.legend{font:10.5px var(--mono);color:var(--mut);margin-top:8px}.legend span{display:inline-block;margin-right:10px}
</style></head><body>
<div class="top">
  <div class="eyebrow">pynative-cost-evaluator · interactive</div>
  <h1>DSv3 内存实验台</h1>
  <p>改并行切分 / 重算策略 → 实时重算峰值 · PP 每 stage 单独看 · 点 timeline 事件看该刻各桶开销。</p>
  <div class="cfg" id="cfg">
    <div class="fld"><label>layers</label><select name="layers"><option>4</option><option selected>8</option></select></div>
    <div class="fld"><label>batch</label><select name="batch"><option selected>1</option><option>2</option></select></div>
    <div class="fld"><label>dp_shard</label><select name="dp"><option>1</option><option selected>2</option><option>4</option><option>8</option></select></div>
    <div class="fld"><label>tp</label><select name="tp"><option selected>1</option><option>2</option><option>4</option></select></div>
    <div class="fld"><label>ep</label><select name="ep"><option selected>1</option><option>2</option><option>4</option><option>8</option></select></div>
    <div class="fld"><label>pp</label><select name="pp"><option selected>1</option><option>2</option><option>4</option></select></div>
    <div class="fld"><label>cp</label><select name="cp"><option selected>1</option><option>2</option></select></div>
    <div class="fld"><label>cp 算法</label><select name="method"><option selected>colossal</option><option>ulysses</option><option>ring</option><option>hybrid</option></select></div>
    <div class="fld"><label>recompute</label><select name="recompute"><option value="None" selected>无</option><option value="full">full</option><option value="select">select</option></select></div>
    <div class="fld"><label>select 模块</label><select name="select"><option value="attn" selected>self_attn(留FFN)</option><option value="mlp">mlp(留attn)</option><option value="both">both(≈full)</option></select></div>
    <div class="kpis">
      <div class="kpi"><div class="n big" id="kpeak">—</div><div class="t">设备峰值（最紧 stage）</div></div>
    </div>
  </div>
</div>
<div class="wrap">
  <div class="tabs" id="tabs"></div>
  <div class="grid">
    <div class="card"><header id="tlhdr">内存时间线（FWD→BWD）</header><div class="b"><p class="desc" id="tldesc"></p><div id="tl"></div>
      <div class="legend" id="leg"></div></div></div>
    <div class="card"><header id="bkhdr">该时刻各桶开销</header><div class="b" id="bk"></div></div>
  </div>
</div>
<script>
const BKC={persistent:"#6b6b6b",act_live:"#4e79a7",kept_frag:"#c0392b",gather_buf:"#59a14f",grad_buf:"#f28e2b",recomp_scratch:"#b07aa1",bwd_scratch:"#e15759",bwd_working_set:"#8cd17d",swap_buf:"#76b7b2",workspace:"#bab0ac",optstep:"#ff9da7",framework:"#d7d7d7"};
const BKD={persistent:"参数+优化器状态",act_live:"存活激活",kept_frag:"B margin(保留-MoE碎片)",gather_buf:"FSDP all-gather",grad_buf:"梯度缓冲",recomp_scratch:"full重算重物化",bwd_scratch:"反向临时(loss fp32)",bwd_working_set:"无重算反向工作集",swap_buf:"激活swap",workspace:"算子workspace",optstep:"优化器step",framework:"框架"};
let cur=null,curStage=0,curEvent=null;
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fmib(m){return m>=1024?(m/1024).toFixed(1)+" GiB":m.toFixed(0)+" MiB";}
function qs(){const o={};document.querySelectorAll("#cfg [name]").forEach(e=>o[e.name]=e.value);return o;}
async function refresh(){
  const params=new URLSearchParams(qs());
  document.body.classList.add("busy");
  const r=await fetch("/api/eval?"+params); const d=await r.json();
  document.body.classList.remove("busy");
  if(!d.ok){document.getElementById("kpeak").textContent="ERR";document.getElementById("tl").innerHTML=`<div class="err">✗ ${esc(d.error)}<br><span style="color:#999">（此并行组合可能不被 mindformers 栈支持 / 触发已知 bug——见 §15 栈限制）</span></div>`;document.getElementById("bk").innerHTML="";document.getElementById("tabs").innerHTML="";return;}
  cur=d; curStage=d.tightest;
  document.getElementById("kpeak").textContent=fmib(d.device_peak);
  // stage 标签
  document.getElementById("tabs").innerHTML=d.stages.map(s=>`<div class="tab ${s.stage===curStage?"on":""} ${s.oom?"oom":""}" data-s="${s.stage}">Stage ${s.stage} · ${fmib(s.peak)}${s.oom?" ⚠OOM":""}</div>`).join("");
  document.querySelectorAll(".tab").forEach(t=>t.addEventListener("click",()=>{curStage=+t.dataset.s;curEvent=null;drawStage();}));
  drawStage();
}
function drawStage(){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("on",+t.dataset.s===curStage));
  const st=cur.stages.find(s=>s.stage===curStage);
  document.getElementById("tlhdr").textContent=`内存时间线 · Stage ${st.stage}（${st.layers}）`;
  document.getElementById("tldesc").innerHTML=`峰值 <b>${fmib(st.peak)}</b> @ ${esc(st.peak_event)}${st.oom?' <span style="color:#c0392b;font-weight:700">⚠ OOM</span>':''} · 点某事件看该刻各桶 ↓`;
  const ev=st.timeline, ymax=Math.max(...ev.map(e=>e.total))*1.02;
  document.getElementById("tl").innerHTML=ev.map((e,i)=>{
    let bars=""; let seg=Object.entries(e.buckets).sort((a,b)=>b[1]-a[1]);
    seg.forEach(([k,v])=>{bars+=`<span style="width:${v/ymax*100}%;background:${BKC[k]};opacity:.85"></span>`;});
    const star=e.total===st.peak?' ★':'';
    return `<div class="tlrow ${i===(curEvent??ev.findIndex(z=>z.total===st.peak))?"on":""}" data-i="${i}"><span class="e">${esc(e.event)}${star}</span><span class="tlbar">${bars}</span><span class="tv">${e.total.toFixed(0)}</span></div>`;
  }).join("");
  document.querySelectorAll(".tlrow").forEach(r=>r.addEventListener("click",()=>{curEvent=+r.dataset.i;drawStage();showBuckets(ev[curEvent]);}));
  document.getElementById("leg").innerHTML=Object.keys(BKC).map(k=>`<span><span style="display:inline-block;width:9px;height:9px;background:${BKC[k]};border-radius:2px"></span> ${k}</span>`).join("");
  showBuckets(ev[curEvent??ev.findIndex(z=>z.total===st.peak)]);
}
function showBuckets(e){
  const bs=Object.entries(e.buckets).sort((a,b)=>b[1]-a[1]); const mx=Math.max(...bs.map(x=>x[1]),1);
  document.getElementById("bkhdr").textContent=`各桶开销 · 事件 ${e.event}（总 ${fmib(e.total)}）`;
  document.getElementById("bk").innerHTML=bs.map(([k,v])=>`<div class="barrow"><span class="bl">${k}</span><span class="bartrack"><span class="barfill" style="width:${v/mx*100}%;background:${BKC[k]}"></span></span><span class="bv">${v.toFixed(0)}·${(v/e.total*100).toFixed(0)}%</span></div><div style="font-size:10.5px;color:#999;margin:-2px 0 4px 126px">${BKD[k]||""}</div>`).join("");
}
document.querySelectorAll("#cfg [name]").forEach(e=>e.addEventListener("change",refresh));
refresh();
</script></body></html>"""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"内存实验台 → http://127.0.0.1:{port}   (Ctrl-C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
