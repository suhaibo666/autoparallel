# 内存评估器审计 + 整改设计（2026-07-06）

> **baseline**: `feat/unified-llm-modelspec` @ `5a5034f`（审计时），后续整改在其上迭代。
> 审计方法：source-faithful，每条断言带已核验 `file:line`。整改总原则：**每项守住 DSv3 锚点
> `validate_dsv3.py` = 12409.5 逐字节 + `pytest` 全绿**（所有整改都在 DSv3 config 不走的路径上，
> 故此不变量必然成立——是每步的硬门）。**docs-first**：每个子代理先补本文档对应 §，再改代码。

## 1. 7 目标达成度（审计结论）

| # | 目标 | 判定 | 关键证据 |
|---|---|---|---|
| 1 | 全/选择/细粒度选择重算 + swap | ✅（细粒度有偏差→D-3） | `mem_timeline.py:376,378,382`；`structure_mem.py:219` |
| 2 | mha/gqa/mla/dsa/hca/csa + dense/moe | ✅（ungated 缺→D-6） | `registry.py:57`；`dsv4_hybrid.py:73`（csa=r4/hca=r128/dsa=r4 indexer 子件） |
| 3 | fsdp2/tp/ep/pp/cp | ⚠️→D-1 | cp 只切参数不切激活（全库 `'cp'` 作 shard key 0 处） |
| 4 | HCCL group 开销 | ⚠️→D-2 | `framework.py:57` 有模型但 `report.py:52` 未接入 |
| 5 | 内存 timeline 图 | ✅ | `timeline_probe.py`（ASCII+PNG）；`record_timeline` |
| 6 | PP 每个 stage | ✅（VPP 过估→D-4） | `mem_timeline.py:305`；`report.py:58` |
| 7 | 配置文件灵活仿真 | ⚠️→D-7 | 对象驱动（LLMConfig+presets），无 yaml 加载器 |

## 2. 发现清单 + 整改决策（用户 2026-07-06 定夺）

| # | 发现 | 决策 | 整改要点 |
|---|---|---|---|
| **D-1** | cp 只切参数、不切激活（over-count） | **修** | cp = **全局域序列切分**：整体激活值都 ÷cp（含 flash-ws）。cp 是独立于 sp 的全局 SP 域 |
| **D-2** | HCCL 未接入报告 | **修** | 按**通信域**估计，接入报告（reserved 预测） |
| **D-3** | 细粒度单 op 选择性重算 under-count | **修** | 细粒度选重**按单个 op 各自估计自己**（每个选中 op 用自身 in/saves 估自己的重算，不依赖段边界已 pin 假设） |
| **D-4** | VPP(v>1) 峰值 over ~V× | **修** | 按 **VPP 理论公式**估 micro 数；整体激活按**实际层数（所有 chunk 层数之和）**估 |
| **D-5** | DSv4-fused 已知 UNDER ~7% | **维持现状** | 融合 kernel 内部量，源码级已定位、已 caveat |
| **D-6** | ungated FFN 硬编码 2*F | **修** | 为 `gated_linear_unit=False` **构建 op 图**（fc1 不 2×） |
| **D-7** | 无配置文件加载器 | **修** | 构建 **mindformers 配置文件 → LLMConfig 转换器** |
| **D-8** | 非整除 PP 层分配堆最后 stage | **修** | 按 mindformers 设计**显式配置每 stage 层数**（从用户 config 读，不自行推测） |
| **D-9** | DSv3 ~0.5% sub-block 残差 | **不处理** | rope cos/sin、norm rstd、cast 临时量 |

## 3. 整改执行顺序（子代理，逐项 docs-first + DSv3 硬门）

1. **并行域核**（同改 `shape_eval/parallel_model/mem_timeline/attention`，故一组内序化）：D-8 → D-1 → D-4。
2. **独立项**（文件不相交，可分批）：D-3（`structure_mem`）、D-2（`framework`+`report`）、D-6（`ffn`+`build_llm`）、D-7（新 `configs/` 转换器）。

各子代理落地机制细节到下方 §4+（每项一节，先写后码）。

## 4. 各项整改机制（子代理填充，source-faithful）

<!-- D-8 / D-1 / D-4 / D-3 / D-2 / D-6 / D-7 各起一节，含 mindformers/Megatron 源码定位 + 公式 + DSv3 复核 -->
