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

**2026-07-25 结构保真修复的影响（band 重钉，CSV 值一字未动）**：此前 UI round-trip 会把现场
yaml 的多个结构字段静默换成 dsv4 预设值（`o_groups` 8→16、`moe_shared_ffn` 2048→3072、
`dsa_indexer_topk` 512→1024、逐层 `compress_ratios` → (0,4,128) 循环、缺 `kv_lora_rank` 补 512），
故本文件此前对账的**并不是这份 yaml 的结构**。保真后：① gather 9084.2（0.866，此前 ~10075/0.960
是替换出来的"准"）；② remat 族过读 2.02×→**1.334×**（真实改善，topk 512 而非 1024）。两处 band
按实测重钉并在各测试 docstring 写明归因。
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
        # kv_lora_rank：现场 yaml **没有**这个键（2026-07-25 round-trip 审计发现）。旧 UI 路径从
        #   dsv4 预设静默补 512 后照常评估，本文件的 CSV 对账就是在该口径下建立的；保真修复后缺它
        #   一律 fail-loud（正确）。为让锚点仍可比，此处**显式补上旧路径替换的同一值**并明示这是
        #   测试补的、不在 yaml 里（dsv4_hybrid 的 op 图不引用该符号——实测 kv_lora∈{1,512,4096}
        #   峰值逐字节相同 → 补值不影响任何桶）。
        mf["model"].setdefault("kv_lora_rank", 512)
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
    # 梯度路径同样取**跨事件最大断面**（与 gather/muon 同法）——CSV 6845 是 stage 瞬态池的逐块
    # 汇总，不是某单个事件的横切。2026-07-25 起必须如此取：`remat_saves` 入账后本 stage 峰事件
    # 从 loss 层 bwd@45 迁到某重算层 bwd@41（真实迁移），而 grad_buf 是**逐层**量、在 loss 层最大
    # → 只看峰事件横切会把 grad 路径读成 6327.2/0.924；跨事件最大 6827.0/0.997 才是与 CSV 可比的
    # 同一物理量（结构未变，只是采样点修正；CSV_GRAD 未动）。
    gpath = max(s["buckets"].get("grad_buf", 0) + s["buckets"].get("grad_accum", 0)
                for s in st7["timeline"])
    return pk["buckets"], gmax, omax, gpath


def test_stage7_gather_regather_matches_csv():
    """① gather re-gather：gather 峰 ≈ CSV 10491（结构量 Σ param_full 补齐 re-gather 窗）。

    **2026-07-25 如实重钉 band (0.92,1.05) → (0.85,1.05)**：round-trip 保真修复后本 stage 的权重
    结构才是 yaml 真值，此前是**被预设替换过**的（更大）结构 —— 逐项实测其对本桶的贡献：
      · `o_groups` 预设 16 而 yaml 真值 **8**  → 旧值多算 **+768 MiB**（分组输出投影 o_group_out）；
      · `moe_shared_expert_intermediate_size` 预设 3072 而真值 **2048** → 多算 **+288 MiB**；
      · `compress_ratios` 预设 (0,4,128) 循环 vs 真表 → **−65 MiB**。
    合计旧口径 Σ param_full ≈ 10075（ratio 0.960，"很准"），真实结构 **9084.2（0.866）**。
    即：此前的吻合有一部分来自结构被替换。CSV 10491 **不动**；欠读 ~13% = 块对齐/小权重 +
    本桶尚未覆盖的 re-gather 成分，需真机逐块 micro-anchor 归因，**不调参掩盖**。
    """
    _, gmax, _, _ = _site_unfused_stage7()
    ratio = gmax / CSV_GATHER
    assert 0.85 <= ratio <= 1.05, (
        f"gather 峰 {gmax:.1f} vs CSV 10491, ratio={ratio:.3f}（结构量=Σ本stage全重算层 param_full;"
        f"残差=块对齐/小权重,报告 §8.5;band 于 2026-07-25 因结构保真修复重钉,见 docstring）")


def test_stage7_muon_overlap_present_and_bounded():
    """② Muon NS 反向重叠：一份 NS workspace(结构量),present 且 ≤ CSV 2264（余为多专家并发,未拟合）。"""
    _, _, omax, _ = _site_unfused_stage7()
    assert omax > 0, "Muon NS 反向重叠桶应 > 0（反向事件叠一份 NS workspace）"
    assert 0.55 <= omax / CSV_MUON <= 1.02, (
        f"Muon 反向重叠 {omax:.1f} vs CSV 2264（一份 NS=68%,余多专家 NS 并发 CSV-文档化,不拟合顶数）")


def test_stage7_grad_path_near_perfect():
    """梯度路径(grad_buf+grad_accum)结构精确——CSV +18 近乎完美,修改不得破坏。"""
    _, _, _, g = _site_unfused_stage7()
    assert abs(g / CSV_GRAD - 1) <= 0.05, f"grad 路径 {g:.1f} vs CSV 6845（应 ±5%）"


def test_stage7_residual_is_framework_gap():
    """三结构桶补齐后,剩余 = csa/indexer fp32 物化框架缺口(激活+重算+bwd_scratch 桶欠 CSV)。
    纯理论口径下此桶不追（框架缺口显式暴露,非模型误差）——守卫其仍 < 真机(欠估方向,OOM 提示)。"""
    b, _, _, _ = _site_unfused_stage7()
    act_recomp_bwd = (b.get("act_live", 0) + b.get("recomp_scratch", 0)
                      + b.get("bwd_scratch", 0) + b.get("bwd_working_set", 0))
    # CSV 激活+重算+bwd = 18373（含未释放 fp32 物化）;理论应显著低（框架缺口）。
    assert act_recomp_bwd < 18373.0, (
        f"激活+重算+bwd 桶 {act_recomp_bwd:.1f} 应 < CSV 18373（差=csa/indexer fp32 框架缺口,纯理论不追）")


def test_stage7_remat_overreads_csv_activation_family_documented():
    """④ **2026-07-25 如实记录的方向翻转**：`remat_saves`（重算再物化的 saved 集）入账后，
    「激活+重算+bwd」族**加上该桶**从欠读 CSV 翻成**过读约 2×**（37146 vs CSV 18373）。

    成因（都不是拟合，全部可溯源）：unfused 分支的 saved 集 census（`dsv4_hybrid.py:187-211`
    的 fp32 复本群 + KL 链）此前是按 **no-recompute/seq2048** 对 185 U1 标定的；在**全重算**下它
    此前对峰值**完全无贡献**（saves 被丢弃、只剩 fml/ci），现在经 `remat` 第一次进入反向峰，而
    其 S² 项在 seq4096 站点配置下达到单层 A−ci≈35.3GB。叠上与 `recomp_scratch`(fml−ci) 的部分
    重叠 → 过读。**按纪律不反向调参**：钉住比值防继续恶化，同时明示这是"过读"（OOM 安全侧但不
    准），待真机逐桶 micro-anchor 决定是否细化 unfused census 的重算态口径 / 扣减 fml 重叠。

    **2026-07-25 重钉过读带 (1.8,2.3) → (1.15,1.6)**：round-trip 保真修复后 `dsa_indexer_topk`
    取 yaml 真值 **512**（此前被 dsv4 预设替换成 **1024**，正好 2×），而 unfused census 的
    `kv_gathered`/`kv_g_fp32`/`attn_weights` 都 ∝ topk → 该族过读从 **2.02× 降到 1.334×**
    （单项实测：topk 512→1024 使本 stage 峰值 +12808 MiB）。**这是过读幅度的真实改善，不是调参**：
    结构对了，比值自然靠近真机；残余 1.33× 仍是上文那条 census 口径问题，继续如实钉住。"""
    b, _, _, _ = _site_unfused_stage7()
    fam = (b.get("act_live", 0) + b.get("recomp_scratch", 0) + b.get("bwd_scratch", 0)
           + b.get("bwd_working_set", 0) + b.get("remat_saves", 0))
    assert fam > 18373.0, "remat 入账后该族应过读 CSV——若回到欠读，说明 remat 被误门控/清零"
    assert 1.15 <= fam / 18373.0 <= 1.6, (
        f"该族 {fam:.1f} / CSV 18373 = {fam/18373.0:.2f}× 越出已记录的过读带 (1.15,1.6)——"
        f"unfused census × remat 的过读幅度漂移,如实重钉并说明理由,勿调参掩盖。")
