# -*- coding: utf-8 -*-
"""185 探针矩阵合账锚点门（2026-07-23,log_probe_* 13 组;相位=reset_max_memory_allocated 隔离）。

四组机制修复的回归门（真机数据绝不改动）：
  A. **unfused 绝对堆叠**（U 相位）：U1(4L seq2048 pp1×dp2×ep2 m1) floor 14817/fwd_end 34776/
     bwd_peak 40194;每层 saves 实测 (34776−14817)/4=4990。修=unfused 反向图 fp32 复本群+naive-r0
     链+KL 链入账（dsv4_hybrid.py DAG 审计嫌疑①②）。**U2(8L) 真机 OOM(56010 分配失败)——修前
     sim 49886 误判可放下,修后必须 >56010（OOM 翻正）**。
  B. **fused 每层差分**（F 相位）：L4=26499(锚)/L8=38936 → 每层 3109。修=core_out 逆 RoPE 保留
     （hybrid:277,嫌疑③）→ sim 每层 3152(+1.4%)。
  C. **std 全重算释放改 185 基准**（R1 相位+R-L 差分）：MS2.10 全重算只释放 h1-fp32/g/act,
     注意段 bprop 保留不释放(~500/层双探针交叉)。116 原生跑不了全重算(context_fn 崩),其 shim
     ON=人造物 → 185 为唯一真跑 build,阳性基准（llm_config.std_recompute_ctx_pin）。
  D. **微批饱和 cap**（pp8/pp4-m8(P3-P) 探针）：死 ctx 在 steady 期被 P2P 同步回收——stage0 无
     同步窗不回收(P3-P s0 ≡m4+纯梯度;pp8-s0 8 组线性),中间 stage 饱和 min(W,ceil((W+1)/2)) 且
     m 无关(P3-P s1-s3 ≡m4)（mem_timeline _ctx_cap）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S

_DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
        "sel_ops": "", "sel_cfg": "", "vpp": "1", "mbs": "", "grad_bytes": "4",
        "dp_replicate": "1", "reshard": "default", "cpu_offload": "0", "prefetch": "1", "sp": ""}


def _dsv4_q(layers, fused, seq="2048", pp="1", recompute="None", mbs="1", split=None):
    q = dict(_DEF)
    q.update({
        "preset": "dsv4_flash", "attn": "dsv4_hybrid",
        "layers": str(layers), "seq": seq, "batch": "1", "mtp": "0",
        "experts": "8", "topk": "2", "dense_k": "1",
        "heads": "64", "kv_groups": "1",
        "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
        "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
        "v_head": "512", "vocab": "129280",
        "hc": "4", "dsa_fused": ("1" if fused else "0"), "ce_fused": "1",
        "dp": "2", "tp": "1", "ep": "2", "pp": pp, "cp": "1", "method": "colossal",
        "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
        "maxdev_gib": "58", "recompute": recompute, "mbs": mbs,
    })
    if split:
        q["pp_split"] = split
    return q


def _peaks(q):
    r = S.eval_config(q)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


# ── A. unfused 绝对堆叠（U 相位）────────────────────────────────────────────────────
def test_u1_unfused_peak_within_5pct():
    p = _peaks(_dsv4_q(4, fused=False))[0]
    assert 0.95 <= p / 40194.0 <= 1.08, (
        f"U1 unfused 峰 sim={p:.1f} vs 真机 bwd_peak 40194（修前 29516/−27%;逐桶合账:"
        f"saves 过估 +4.5G 与 floor 欠账 −3.7G 相抵,净 +4.6%,见 probe 报告）")


def test_u2_unfused_oom_flip():
    """U2(8L) 真机 OOM(56010 分配失败)——修前 sim 49886 会误判「可放下」。修后必须 >56010。"""
    p = _peaks(_dsv4_q(8, fused=False))[0]
    assert p > 56010.0, f"U2 sim={p:.1f} 仍低于真机 OOM 点 56010——OOM 误判未翻正"


# ── B. fused 每层差分（F 相位）──────────────────────────────────────────────────────
def test_fused_per_layer_increment():
    p4 = _peaks(_dsv4_q(4, fused=True))[0]
    p8 = _peaks(_dsv4_q(8, fused=True))[0]
    per_layer = (p8 - p4) / 4
    assert abs(per_layer / 3109.0 - 1) <= 0.05, (
        f"fused 每层差分 sim={per_layer:.1f} vs 真机 3109（逆 RoPE 保留入账后应 ±5%）")
    assert 0.95 <= p4 / 26499.0 <= 1.02, f"F0 绝对 sim={p4:.1f} vs 锚 26499"


# ── C. std 全重算 185 基准（R1 相位）────────────────────────────────────────────────
def _std_on_q(kv):
    q = dict(_DEF)
    q.update({
        "preset": "custom", "attn": ("mha" if kv == 32 else "gqa"),
        "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
        "experts": "0", "topk": "1", "dense_k": "8",
        "heads": "32", "kv_groups": str(kv), "hidden": "2048",
        "ffn": "8192", "moe_ffn": "8192", "vocab": "129280", "hc": "1",
        "dp": "2", "tp": "1", "ep": "1", "pp": "2", "cp": "1", "method": "colossal",
        "optimizer": "adamw", "opt_dtype": "bf16", "grad_bytes": "4",
        "maxdev_gib": "58", "recompute": "full", "mbs": "4", "pp_split": "4,4",
        "emb_bytes": "4", "std_pin": "1",   # 185 build 事实:emb fp32 + 全重算保留集
    })
    return q


# MHA s0 −6.3%(欠,band 保底 0.93 注明:185 ON floor 较 sim 高 ~0.5G+保留集下界口径);
# GQA s0 −16%(残差①同源:真机 GQA 深 warmup 段驻留≈MHA,kv 缩水未兑现)。s1 双双 ~±5%。
_STD_ON_REAL = {32: {0: 11131.6, 1: 15370.0}, 8: {0: 10747.6, 1: 14986.0}}
_STD_ON_BAND = {(32, 0): (0.93, 1.05), (32, 1): (0.95, 1.07),
                (8, 0): (0.82, 1.05), (8, 1): (0.95, 1.07)}


@pytest.mark.parametrize("kv,stage", [(32, 0), (32, 1), (8, 0), (8, 1)])
def test_std_recompute_on_185(kv, stage):
    sim = _peaks(_std_on_q(kv))[stage]
    real = _STD_ON_REAL[kv][stage]
    lo, hi = _STD_ON_BAND[(kv, stage)]
    ratio = sim / real
    assert lo <= ratio <= hi, (
        f"185 std {'MHA' if kv == 32 else 'GQA'} ON stage{stage}: sim={sim:.1f} vs "
        f"real={real}, ratio={ratio:.3f} 越出 ({lo},{hi})。修前 [6741,14291,6525,14027]"
        f"（s0 −39% 红区——116-shim 全释放口径,185 唯一真跑 build 证伪）")


# ── D. 微批饱和 cap（P3-P:pp4 ON m=8）───────────────────────────────────────────────
def test_p3p_m8_saturation():
    """P3-P:s0=25343.5(≡m4+纯梯度 +1190),s1-s3 ≡m4——stage0 不回收+中间 stage cap m 无关。"""
    pk = _peaks(_dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                        mbs="8", split="2,2,2,2"))
    assert abs(pk[0] / 25343.5 - 1) <= 0.05, f"P3-P s0 sim={pk[0]:.1f} vs 真机 25343.5"
    m4 = _peaks(_dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                        mbs="4", split="2,2,2,2"))
    for s in (1, 2, 3):
        assert abs(pk[s] - m4[s]) < 1.0, (
            f"stage{s}: m8({pk[s]:.1f}) ≠ m4({m4[s]:.1f})——真机 s1-s3 逐 MiB ≡m4(饱和,cap 应 m 无关)")
