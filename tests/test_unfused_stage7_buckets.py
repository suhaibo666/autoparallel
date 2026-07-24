# -*- coding: utf-8 -*-
"""unfused DSv4-Flash stage7 三结构桶对 CSV 逐块实测锚（2026-07-24，报告 §8.5）。

真机 CSV(`memory_block_sync_fixed.csv`,UTF-16,63860 块,带 python_stack 归属;末 stage7 瞬态池)：
这三桶是**评估器可建准的结构量**(非框架缺口——框架缺口=csa/indexer fp32 物化,保持纯理论清零)。
逐块对账见报告 §8.5 表。三处修复(mem_timeline.py,仅 pp>1 全重算反向 / Muon 激活)：
  ① FSDP re-gather(gather_buf):理论前向 2-unit 口径 5510 → CSV 10491(`param.py:519`,102 块)。
     修=全重算反向叠 Σ本stage全重算层 param_full(recompute-forward 整段 re-gather)。
  ② Muon NS 反向重叠(optstep 桶):理论峰外 0 → CSV 2264(`muon.py`,877 块)。
     修=反向事件叠一份最大矩阵 NS workspace(_MUON_NS_WORKSPACE_MULT×分片×4)。
  ③ 全重算反向工作集(bwd_working_set):理论 bwd_scratch 4040 → CSV autograd 7366+linear 269。
     修=full-recompute 非 loss 层 max(0,fml−bwd_scratch)。s7 峰在 loss 层→此桶该事件 0(不双算),
     该桶在中部 transformer/MoE 层 stage 生效(见 pp8/s0)。
纪律：真机 CSV 值绝不改动;结构量修后应更接近真机,剩余=框架缺口显式暴露。
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import serve_explorer as S
from cost_eval.configs.from_mindformers import from_mindformers_dict

_SITE = r"C:\Users\suhaibo\xwechat_files\suhaibo1993_9a8e\msg\file\2026-07\test.yaml"
_DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
        "sel_ops": "", "vpp": "1", "mbs": "", "grad_bytes": "4"}

# CSV 逐块实测锚（报告 §8.5，绝不改动）
CSV_GATHER = 10491.0      # param.py:519, 102 块
CSV_MUON = 2264.0         # muon.py, 877 块
CSV_GRAD = 6845.0         # param.py:862（近乎完美 +18）


def _site_unfused_stage7():
    if not os.path.exists(_SITE):
        pytest.skip("现场 test.yaml 不可用（本机私有路径）")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = yaml.safe_load(open(_SITE, encoding="utf-8"))
        mf["model"]["apply_dsa_kernel_fusion"] = False
        mf2, _ = S._mf_adapt(mf)
        w = []
        S._materialize_nested_offset(mf2, w)
        fields = S._bundle_to_fields(from_mindformers_dict(mf2))
        q = dict(_DEF)
        q.update({k: str(v) for k, v in fields.items() if v is not None})
        r = S.eval_config(q)
    assert r["ok"], r.get("errors")
    st7 = max(r["stages"], key=lambda s: s["stage"])
    pk = max(st7["timeline"], key=lambda x: x["total"])
    # gather / muon(optstep during bwd) / grad 取该配置反向阶段的最大断面（跨事件）
    gmax = max(s["buckets"].get("gather_buf", 0) for s in st7["timeline"])
    omax = max(s["buckets"].get("optstep", 0) for s in st7["timeline"]
               if s["event"].startswith("bwd@"))
    return pk["buckets"], gmax, omax


def test_stage7_gather_regather_matches_csv():
    """① gather re-gather：修后 gather 峰 ≈ CSV 10491（结构量 Σ param_full 补齐 re-gather 窗）。"""
    _, gmax, _ = _site_unfused_stage7()
    ratio = gmax / CSV_GATHER
    assert 0.92 <= ratio <= 1.05, (
        f"gather 峰 {gmax:.1f} vs CSV 10491, ratio={ratio:.3f}（结构量=Σ本stage全重算层 param_full;"
        f"残差=块对齐/小权重,报告 §8.5）")


def test_stage7_muon_overlap_present_and_bounded():
    """② Muon NS 反向重叠：一份 NS workspace(结构量),present 且 ≤ CSV 2264（余为多专家并发,未拟合）。"""
    _, _, omax = _site_unfused_stage7()
    assert omax > 0, "Muon NS 反向重叠桶应 > 0（反向事件叠一份 NS workspace）"
    assert 0.55 <= omax / CSV_MUON <= 1.02, (
        f"Muon 反向重叠 {omax:.1f} vs CSV 2264（一份 NS=68%,余多专家 NS 并发 CSV-文档化,不拟合顶数）")


def test_stage7_grad_path_near_perfect():
    """梯度路径(grad_buf+grad_accum)结构精确——CSV +18 近乎完美,修改不得破坏。"""
    b, _, _ = _site_unfused_stage7()
    g = b.get("grad_buf", 0) + b.get("grad_accum", 0)
    assert abs(g / CSV_GRAD - 1) <= 0.05, f"grad 路径 {g:.1f} vs CSV 6845（应 ±5%）"


def test_stage7_residual_is_framework_gap():
    """三结构桶补齐后,剩余 = csa/indexer fp32 物化框架缺口(激活+重算+bwd_scratch 桶欠 CSV)。
    纯理论口径下此桶不追（框架缺口显式暴露,非模型误差）——守卫其仍 < 真机(欠估方向,OOM 提示)。"""
    b, _, _ = _site_unfused_stage7()
    act_recomp_bwd = (b.get("act_live", 0) + b.get("recomp_scratch", 0)
                      + b.get("bwd_scratch", 0) + b.get("bwd_working_set", 0))
    # CSV 激活+重算+bwd = 18373（含未释放 fp32 物化）;理论应显著低（框架缺口）。
    assert act_recomp_bwd < 18373.0, (
        f"激活+重算+bwd 桶 {act_recomp_bwd:.1f} 应 < CSV 18373（差=csa/indexer fp32 框架缺口,纯理论不追）")
