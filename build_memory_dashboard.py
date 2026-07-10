"""整网内存全景面板 → 自包含 HTML（纯 Python，零依赖）。

三段：① 模型结构逐层内存（embedding→N 层→head，每层 pinned 激活 + 参数）；② 整网内存时间线
（FWD→BWD 逐事件 8+ 桶堆叠面积 + total 折线 + 峰值 ★）；③ 峰值时刻各桶大小（横向柱 + 占比）。

用法：python build_memory_dashboard.py            # DSv3 8L 无重算 dp=2
     SIM_SELECT=attn python build_memory_dashboard.py   # select self_attn（看 kept_frag margin）
输出：analysis/viz/memory_dashboard.html
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from cost_eval.opdag.render import timeline_to_svg, _esc

MiB = 2 ** 20
_LAYER_COLOR = {"embedding": "#9c9c9c", "mla_dense": "#4e79a7", "mla_moe": "#2f4b7c",
                "dense": "#4e79a7", "moe": "#2f4b7c", "lm_head": "#e15759"}
_BUCKET_COLOR = {
    "persistent": "#6b6b6b", "act_live": "#4e79a7", "kept_frag": "#c0392b",
    "gather_buf": "#59a14f", "grad_buf": "#f28e2b", "recomp_scratch": "#b07aa1",
    "bwd_scratch": "#e15759", "bwd_working_set": "#8cd17d", "swap_buf": "#76b7b2",
    "workspace": "#bab0ac", "optstep": "#ff9da7", "framework": "#d7d7d7",
}
_BUCKET_DESC = {
    "persistent": "参数 + 优化器状态（常驻）", "act_live": "存活激活（saved，fwd→bwd 驻留）",
    "kept_frag": "B 标定 margin（保留-MoE 碎片长尾）", "gather_buf": "FSDP all-gather 整层权重缓冲",
    "grad_buf": "参数梯度缓冲", "recomp_scratch": "full 重算重物化", "bwd_scratch": "反向临时（loss probs fp32 等）",
    "bwd_working_set": "无重算反向工作集", "swap_buf": "激活 swap H2D", "workspace": "算子 workspace",
    "optstep": "优化器 step 瞬态", "framework": "框架常驻",
}


def per_layer_saved(timeline, spec):
    """从 FWD 事件的 act_live 增量抽每层 pinned 激活。返回 [(lid, name, saved_MiB)]。

    注意 mem_timeline 在 **pin 该层 saved 之前** rec(`fwd:lid`)（gather_buf 采样点），故 `fwd:lid`
    的 act_live = 层 0..lid-1 之和。层 lid 的 saved = 下一事件 act_live − 本事件（末层用 `fwd_end` 收尾）。
    """
    pat = list(getattr(spec, "layer_pattern", []))
    fwd = sorted((int(s.event.split(":")[1]), s.breakdown.act_live) for s in timeline
                 if s.event.startswith("fwd:"))
    end = next((s.breakdown.act_live for s in timeline if s.event == "fwd_end"), None)
    avs = [av for _, av in fwd] + ([end] if end is not None else [fwd[-1][1] if fwd else 0])
    rows = []
    for i, (lid, _) in enumerate(fwd):
        name = pat[lid] if 0 <= lid < len(pat) else f"L{lid}"
        rows.append((lid, name, max(0, avs[i + 1] - avs[i]) / MiB))
    return rows


def bar_row(label, val, maxv, color, sub=""):
    w = 0 if maxv == 0 else max(1.5, val / maxv * 100)
    return (f'<div class="row"><div class="lbl">{_esc(label)}<span class="sub">{_esc(sub)}</span></div>'
            f'<div class="track"><div class="fill" style="width:{w:.1f}%;background:{color}"></div>'
            f'<span class="val">{val:,.0f} MiB</span></div></div>')


def build(N=8, select=None):
    spec, d, fl = build_dsv3_spec(N)
    d.B = 1
    if select == "attn":
        ops = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, N + 1)}
        rc, mode_lbl = RecomputeSpec("select", select_ops=ops), "select self_attn（重算 attn、留 FFN）"
    else:
        rc, mode_lbl = RecomputeSpec("None"), "无重算（all saved）"
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    rep = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                    HardwareSpec(max_device_memory=64 * 2 ** 30, framework_reserve=0),
                    rc, SwapSpec()).evaluate(record_timeline=True)
    sp = rep.per_stage[rep.tightest_stage]
    peak = sp.peak_bytes / MiB
    bd = sp.breakdown

    # ① 逐层激活
    layers = per_layer_saved(sp.timeline, spec)
    maxl = max((v for _, _, v in layers), default=1)
    layer_html = "".join(
        bar_row(name, v, maxl, _LAYER_COLOR.get(name, "#888"), sub=f"L{lid}")
        for lid, name, v in layers if v > 0.05)

    # ③ 峰值各桶
    buckets = [(k, getattr(bd, k, 0) / MiB) for k in _BUCKET_COLOR if getattr(bd, k, 0)]
    buckets.sort(key=lambda x: -x[1])
    maxb = max((v for _, v in buckets), default=1)
    bucket_html = "".join(
        bar_row(k, v, maxb, _BUCKET_COLOR[k], sub=f"{v/peak*100:.0f}% · {_BUCKET_DESC.get(k,'')}")
        for k, v in buckets)

    # ② 时间线
    tl_svg = timeline_to_svg(sp.timeline, title=f"整网内存时间线 · DSv3 {N}L · {mode_lbl}")

    total_act = sum(v for _, _, v in layers)
    return _PAGE.format(
        N=N, mode=_esc(mode_lbl), peak=f"{peak:,.0f}", event=_esc(sp.peak_event),
        persistent=f"{bd.persistent/MiB:,.0f}", act=f"{total_act:,.0f}", nlayers=len(layers),
        layer_rows=layer_html, timeline=tl_svg, bucket_rows=bucket_html)


_PAGE = """<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#171b24;--mut:#68707e;--line:#e4e7ec;--blue:#2f4b7c;--saved:#c0392b;--mono:ui-monospace,Consolas,Menlo,monospace}}
*{{box-sizing:border-box}}body{{margin:0}}
.wrap{{max-width:1200px;margin:0 auto;padding:38px 26px 72px;color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
.eyebrow{{font:600 12px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--blue)}}
h1{{font-size:29px;margin:11px 0 6px;letter-spacing:-.01em;text-wrap:balance}}
.lede{{color:var(--mut);margin:0 0 20px;max-width:74ch}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px}}
.kpi{{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px 16px}}
.kpi .n{{font:700 24px/1 var(--mono);letter-spacing:-.02em}}
.kpi .n.big{{color:var(--saved)}}
.kpi .t{{color:var(--mut);font-size:12.5px;margin-top:5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-top:24px;overflow:hidden}}
.card>header{{padding:16px 20px 12px;border-bottom:1px solid var(--line)}}
.card h2{{margin:0;font-size:17px;letter-spacing:-.01em}}
.card header p{{margin:3px 0 0;color:var(--mut);font-size:13px}}
.body{{padding:14px 20px 18px}}
.row{{display:flex;align-items:center;gap:12px;padding:4px 0}}
.lbl{{width:150px;flex-shrink:0;font:13px/1.3 var(--mono);text-align:right}}
.lbl .sub{{display:block;color:var(--mut);font-size:10.5px}}
.track{{position:relative;flex:1;height:24px;background:#f0f2f5;border-radius:5px;overflow:hidden}}
.fill{{height:100%;border-radius:5px;opacity:.85}}
.val{{position:absolute;right:9px;top:3px;font:600 12px/1 var(--mono);color:#333;font-variant-numeric:tabular-nums}}
.scroll{{overflow-x:auto;padding:12px}}.scroll svg{{display:block;height:auto}}
.hint{{color:var(--mut);font-size:12.5px;margin:10px 20px 0}}
</style>
<div class="wrap">
  <div class="eyebrow">pynative-cost-evaluator · 整网内存全景</div>
  <h1>DSv3 {N}L 显存分解面板</h1>
  <p class="lede">{mode} · dp_shard=2 · pp=1 · seq=4096 · B=1。逐层激活、整网内存时间线、峰值各桶——一屏看清显存去哪了。</p>
  <div class="kpis">
    <div class="kpi"><div class="n big">{peak} MiB</div><div class="t">设备峰值 @ {event}</div></div>
    <div class="kpi"><div class="n">{persistent} MiB</div><div class="t">persistent（参数+优化器状态）</div></div>
    <div class="kpi"><div class="n">{act} MiB</div><div class="t">存活激活合计（{nlayers} 层）</div></div>
  </div>

  <div class="card"><header><h2>① 模型结构 · 逐层存活激活</h2><p>每层 FWD 时 pin 进 act_live 的激活（embedding → transformer 层 → head），宽度 ∝ MiB。</p></header>
    <div class="body">{layer_rows}</div>
    <p class="hint">MLA≈94 MiB/层（含 q/kv layernorm fp32 24+8）、MoE-FFN≈152 MiB/层（grouped-GEMM 中间量）——per-Cell op-DAG 提取交叉验证。</p>
  </div>

  <div class="card"><header><h2>② 整网内存时间线（FWD → BWD）</h2><p>每个事件（逐层前向 / 逐层反向 / 优化器 step）的 12 桶堆叠 + total 折线，峰值 ★。</p></header>
    <div class="scroll">{timeline}</div>
  </div>

  <div class="card"><header><h2>③ 峰值时刻 · 各桶大小</h2><p>峰值事件那一刻各内存桶的字节 + 占峰值比例。</p></header>
    <div class="body">{bucket_rows}</div>
  </div>
</div>"""


if __name__ == "__main__":
    N = int(os.environ.get("SIM_LAYERS", "8"))
    select = os.environ.get("SIM_SELECT")  # "attn" → select self_attn；否则无重算
    html = build(N=N, select=select)
    os.makedirs("analysis/viz", exist_ok=True)
    out = "analysis/viz/memory_dashboard.html"
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, os.path.getsize(out), "bytes")
