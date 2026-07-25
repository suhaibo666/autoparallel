# -*- coding: utf-8 -*-
"""185 探针矩阵合账锚点门（2026-07-23,log_probe_* 13 组;相位=reset_max_memory_allocated 隔离）。

四组机制修复的回归门（真机数据绝不改动）：
  A. **unfused 绝对堆叠**（U 相位）：U1(4L seq2048 pp1×dp2×ep2 m1) floor 14817/fwd_end 34776/
     bwd_peak 40194;每层 saves 实测 (34776−14817)/4=4990。修=unfused 反向图 fp32 复本群+naive-r0
     链+KL 链入账（dsv4_hybrid.py DAG 审计嫌疑①②）。**U2(8L) 真机 OOM(56010 分配失败)——修前
     sim 49886 误判可放下,修后必须 >56010（OOM 翻正）**。
  B. **fused 每层差分**（F 相位）：L4=26499(锚)/L8=38936 → 每层 3109。修=core_out 逆 RoPE 保留
     （hybrid:277,嫌疑③）→ sim 每层 3152(+1.4%)。
  C. **std 全重算（2026-07-24 口径切换：纯理论）**：全重算只留层入口 checkpoint_input,注意段
     bprop 张量不再 pin（去 std_recompute_ctx_pin 经验保留集）。真机全重算不释放注意段 bprop 保留
     (~500/层双探针交叉)——理论 < 真机的差距为**框架释放缺口**,双断言(理论值+缺口)。
  D. **pp4 ON m=8（纯理论：去经验饱和 cap）**：saved=checkpoint_input 与 m 无关（结构性质,非 cap
     造出的）→ s1-s3 仍 m 无关;s0 理论 pin 住。真机 25343.5 与理论差=框架缺口。
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


# ── C. std 全重算 **纯理论**（2026-07-24 口径切换：去 std_recompute_ctx_pin 经验保留集）─────
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
        "emb_bytes": "4",   # 185 build 事实:emb fp32（std_pin 保留集已随口径切换删除）
    })
    return q


_STD_ON_REAL = {32: {0: 11131.6, 1: 15370.0}, 8: {0: 10747.6, 1: 14986.0}}
# 纯理论值（全重算只留层入口 checkpoint_input，注意段 bprop 张量不再 pin）。理论 << 真机,
#   差距=框架释放缺口（MS2.10 全重算实际不释放注意段 bprop 保留，~500/层，双探针交叉背书）。
# 2026-07-24 §8.5 三结构桶(pp2>1 全重算反向:re-gather/bwd_ws;AdamW→无 Muon 项)入账后上移。
#   理论仍 < 真机,差距=注意段 bprop 保留框架缺口(缩小)。
# 2026-07-25 `remat_saves`(重算再物化的 saved 集,= max(0,A−ci))入账后 **s0 再上移**:
#   MHA(kv=32) s0 7362.8→8074.8(A−ci=712.0 = std MHA 层 saves 去 ci)、GQA(kv=8) s0 7026.8→7594.8
#   (A−ci=568.0);**s1 逐 MiB 不变**——尾 stage 峰在 loss/head 层(非重算层)的 bwd 事件,该事件此桶 0
#   (真实,不是被门挡掉)。方向:MHA s0 0.661→0.725、GQA s0 0.655→0.707,四点仍全部 sim < 真机 → 缺口
#   **收窄**,框架缺口方向不变。
_STD_ON_THEO = {32: {0: 8074.8, 1: 14802.8}, 8: {0: 7594.8, 1: 14490.8}}


@pytest.mark.parametrize("kv,stage", [(32, 0), (32, 1), (8, 0), (8, 1)])
def test_std_recompute_on_185_theoretical(kv, stage):
    """(a) 纯理论值 pin 住防漂移（全重算只留 checkpoint_input）。"""
    sim = _peaks(_std_on_q(kv))[stage]
    theo = _STD_ON_THEO[kv][stage]
    assert abs(sim - theo) < 0.5, (
        f"185 std {'MHA' if kv == 32 else 'GQA'} ON stage{stage} 理论漂移: sim={sim:.1f} vs {theo:.1f}")


@pytest.mark.parametrize("kv,stage", [(32, 0), (32, 1), (8, 0), (8, 1)])
def test_std_recompute_on_185_framework_gap(kv, stage):
    """(b) 框架缺口：真机全重算不释放注意段 bprop 保留(~500/层)——理论 < 真机,差距为框架缺口。"""
    sim = _peaks(_std_on_q(kv))[stage]
    real = _STD_ON_REAL[kv][stage]
    assert sim < real, (
        f"185 std {'MHA' if kv == 32 else 'GQA'} ON stage{stage}: 理论 {sim:.1f} 应 < 真机 {real}"
        f"（框架释放缺口={real-sim:.0f}MiB，注意段 bprop 保留 MS 全重算不释放，非模型误差）。")


# ── D. pp4 ON m=8 **纯理论**（去经验饱和 cap）────────────────────────────────────────
def test_p3p_m8_theoretical_and_gap():
    """纯理论口径去经验饱和 cap（cap 是死 ctx 的经验回收模型；纯理论无免疫 ctx 可回收）。
    s0 理论 pin 住；中间 stage s1-s3 仍 **m 无关**（saved=checkpoint_input 与 m 无关，是结构性质，
    非 cap 造出的）。s0 理论 << 真机 25343.5——差距=框架释放缺口。"""
    pk = _peaks(_dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                        mbs="8", split="2,2,2,2"))
    # 2026-07-24 §8.5 三桶;2026-07-25 `remat_saves` 入账 15207.9 → 18783.7（+3575.8 = 该 stage
    #   峰值重算层的 A−ci）。s0 vs 真机 25343.5：0.600 → 0.741，**收窄**且仍欠读（缺口方向不变）。
    assert abs(pk[0] - 18783.7) < 0.5, f"P3-P s0 理论漂移 sim={pk[0]:.1f} vs 18783.7"
    assert pk[0] < 25343.5, f"P3-P s0 理论 {pk[0]:.1f} 应 < 真机 25343.5（框架缺口={25343.5-pk[0]:.0f}MiB）"
    m4 = _peaks(_dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                        mbs="4", split="2,2,2,2"))
    for s in (1, 2, 3):
        assert abs(pk[s] - m4[s]) < 1.0, (
            f"stage{s}: m8({pk[s]:.1f}) ≠ m4({m4[s]:.1f})——中间 stage saved=ci 应 m 无关")
