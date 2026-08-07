# Target Design v2 P1 PyNative CodeIR Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 `docs/target-design-v2` 从 `π₀ + pass/rewrite` 架构修订为“当前源码与配置直接生成硬件无关 `CodeIR(P)`，再展开 `RuntimeEventPlan`”的 PyNative 工程估算器设计，并把 raw trace 固定为测试专用 oracle。

**Architecture:** 生产链采用 `SourceSnapshot + ModelSpec + P + CompileEnvFacts → CodeIR(P) → RuntimeEventPlan → HardwareProfile/CalibrationSet 绑定 → SimulationPlan → MemoryEventView/TimeEventView`。`CodeIR` 不可变、无硬件字段、无 compiler fusion 或图替换；训练派生行为通过引用 `code_node_id` 的事件实例表达。验证链独立使用 `TraceFixture → IRConformance`，不进入生产 API、digest 或缓存键。

**Tech Stack:** HTML 设计模板、Python 3 标准库门禁/构建脚本、`unittest`、`pytest`、本地浏览器视觉检查。

**Approved design:** `docs/superpowers/specs/2026-08-07-target-design-v2-p1-pynative-codeir-design.md`

**Workspace constraint:** 目标模板、门禁、HANDOFF 和生成 HTML 在本轮开始前已有未提交修改。保留这些修改，不 checkout/reset，不把它们作为独立 commit 提交；本计划只在当前工作区增量修订并交付最终 diff。开发者禁止本轮委派，因此计划由主代理顺序执行。

---

### Task 1: 先把 P1 契约写进 verifier，并证明旧模板会失败

**Files:**
- Modify: `docs/target-design-v2/tools/verify_gates.py`
- Modify: `docs/target-design-v2/tools/test_verify_gates.py`
- Read-only baseline: `docs/target-design-v2/src/index.template.html`

**Step 1: 扩展 verifier 单元测试**

把 verifier 的纯检查入口整理为可接收 HTML 字符串的函数，并新增 `VerifyGatesContractTest`，针对 P1 正向/反向词项以及 trace 边界至少覆盖：

- 删除一个 `CodeIR(P)` 等必需契约时 verifier 失败；
- 注入旧 `π₀` 权威图、`Pass_impl_select`、`GraphVersion` rewrite 等旧语义时 verifier 失败；
- 模板若声称 trace 进入 `model_digest`、缓存键或生产输入时 verifier 失败；
- GBK 控制台兼容性测试继续保留。

**Step 2: 运行测试，确认测试先失败**

Run:

```powershell
cd docs/target-design-v2
python -m unittest tools.test_verify_gates.VerifyGatesContractTest -v
```

Expected: 新测试因 verifier 尚未提供 P1 检查入口或未识别旧语义而失败。

**Step 3: 更新 verifier 契约**

使 `REQUIRED_TEXT` 至少检查：

```text
目标方案设计 v4.1
CodeIR(P)
RuntimeEventPlan
SimulationPlan
EffectSummary
CompileEnvFacts
HardwareProfile
SemanticValue
TraceFixture
IRConformance
```

使旧架构/越界语义检查至少覆盖：`CoreIR = PE(Source, ModelSpec, EnvFacts, π₀)`、策略单位元权威图、`Pass_impl_select`、实现选择与融合、通用 replacement/rewrite 链、trace 作为预测输入、trace 进入 digest/cache。保留 P0 的旧产品语义禁用项和 15 章/18 门结构检查。

**Step 4: 运行单元测试**

Run:

```powershell
python -m unittest tools.test_verify_gates.VerifyGatesContractTest -v
```

Expected: verifier 的纯契约单元测试通过。此时不运行整套 console integration test，因为它仍会针对尚未修订的真实模板返回非零。

**Step 5: 对旧模板运行新 verifier，记录预期失败**

Run:

```powershell
python tools/verify_gates.py
```

Expected: 非零退出，至少报告缺少 P1 必需契约和残留旧架构语义。这是模板修订前的 RED 检查点。

---

### Task 2: 重写权威模板的 IR 与输入边界

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`

**Step 1: 修订标题、产品边界与总数据流**

把标题升级为 `目标方案设计 v4.1`。保留 P0 工程估算器、logical allocated peak、DES step time 和同口径配置比较契约，替换旧主链为：

```text
SourceSnapshot + ModelSpec + normalized config P + CompileEnvFacts
  → CodeIR(P)
  → RuntimeEventPlan
  → hardware binding + cost evaluation
  → SimulationPlan
  → MemoryEventView / TimeEventView
```

**Step 2: 拆分环境输入**

明确：

- `CompileEnvFacts` 只含框架/运行时版本、source feature flags、静态 import/config facts；
- `HardwareProfile` 含设备、容量、拓扑、带宽、吞吐和 allocator/alignment 常量；
- `CalibrationSet` 是可选、版本化成本数据；
- `TraceFixture` 不属于生产输入。

声明硬件变化不得改变 `CodeIR` 与 `model_digest`。

**Step 3: 替换 `π₀` 与 pass/rewrite 章节**

明确每个当前配置 `P` 直接从源码求值得到 `CodeIR(P)`：

- 不从策略单位元图重建 TP/MoE/remat 分支；
- 源码显式 fused op 保持单节点，普通多 op 保持多节点；
- 不设计 compiler fusion、`Pass_impl_select`、replacement、superseded 或 `GraphVersion`；
- `CodeIR` 创建后不可变。

删除不兼容的 `{{SVG:05-identities}}`、`{{SVG:06-placement-execution}}` 引用；保留的图必须与新契约一致。

**Step 4: 定义 `RuntimeEventPlan`**

描述 forward/backward/optimizer/recompute/communication/microbatch/pipeline 事件展开，并规定重计算生成新的 `ExecEvent`、引用相同 `code_node_id`，不复制或替换 `CodeIR` 节点。硬件 stream/resource 只在 `SimulationPlan` 绑定。

**Step 5: 运行 P1 词项扫描**

Run:

```powershell
rg -n "π₀|Pass_impl_select|实现选择与融合|replacement|只增不改|GraphVersion|CodeIR\(P\)|RuntimeEventPlan|HardwareProfile" src/index.template.html
```

Expected: 新架构词项存在；旧架构词项不存在，除非是明确且不会触发 verifier 的历史对照说明。

---

### Task 3: 补全 Effect、逐表达式证据与测试专用 trace 契约

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`

**Step 1: 加入 `EffectSummary`**

覆盖 `reads/writes/mutates/rng/collective/host_side_effect/replayable/deterministic/reentrant`。说明 effect 缺失或冲突会阻断受影响的重放、重排或并发，不以默认纯函数继续；不把 effect 用于不存在的融合变换。

**Step 2: 加入 `SemanticValue<T>`**

让 shape、dtype、storage/alias、autograd、placement、resource/cost、workspace、effects 分别携带 `value/quality/evidence_ref/assumptions/coverage_tags`。明确 Native 只代表来源，不使全部语义面自动 exact。

**Step 3: 定义 `IRConformance` 测试链**

区分：

- `CodeIR` 一致性：op identity/count/order、依赖、shape/dtype、配置分支、源码映射；
- `RuntimeEventPlan` 一致性：backward/remat/optimizer、collective、rank/device/stream/timestamp、microbatch/pipeline；
- 默认采集使用标准 profiler/memory tracker 和独立 runner；业务源码或框架改动不是必需条件；可选 hook/marker 必须 A/B 检查扰动。

**Step 4: 锁定 trace 负边界**

明确 raw `TraceFixture`：只用于测试、不得进入生产 API/`SpecBundle`/`model_digest`/缓存键/coverage/confidence，缺失不得改变生产预测。说明 `CalibrationSet` 与 raw trace 的差异，禁止把待预测运行 trace 偷渡为标定输入。

**Step 5: 更新 18 道结构门**

保持门编号集合不变：

- `G-IR1..3` 改为 CodeIR 来源/硬件独立、event/effect/逐表达式证据完整性；
- `G-MEM1..6` 保持 logical allocated lifecycle 不变量；
- `G-TIME1..6` 保持 event DAG、stream/resource/collective/DES 不变量；
- `G-REP1..3` 覆盖报告完整性、同口径比较与 trace/IRConformance 的测试专用边界。

**Step 6: 运行 verifier**

Run:

```powershell
python tools/verify_gates.py
```

Expected: 10+ P1/P0 必需契约、全部禁用语义、15 章和恰好 18 道门全部通过。

---

### Task 4: 更新 HANDOFF，并重建两份 HTML

**Files:**
- Modify: `docs/target-design-v2/HANDOFF.md`
- Regenerate: `docs/target-design-v2/index.html`
- Regenerate: `docs/target-design-v2/artifact.html`

**Step 1: 重写 HANDOFF 架构摘要**

保留 P0 输出与后端公式，替换旧 `CoreIR → pass → SchedIR` 主线；加入：

- `CodeIR(P)` 当前配置直接求值；
- runtime event expansion；
- CodeIR 硬件无关与不可变；
- Effect/SemanticValue；
- trace-only validation 与 CalibrationSet 分界；
- 新 18 门含义；
- 工作区和权威源说明。

同时将设计依据列为 P0 与本 P1 两份 approved spec。

**Step 2: 构建文档**

Run:

```powershell
python tools/build_doc.py
```

Expected: 成功生成 `index.html` 与 `artifact.html`，无未替换占位符。

**Step 3: 校验模板和生成物一致性**

Run:

```powershell
python tools/verify_gates.py
python -m unittest discover -s tools -p "test_*.py" -v
```

Expected: 全部通过。

---

### Task 5: 结构、回归与视觉验收

**Files:**
- Verify: `docs/target-design-v2/src/index.template.html`
- Verify: `docs/target-design-v2/index.html`
- Verify: `docs/target-design-v2/artifact.html`
- Verify: repository test suite

**Step 1: 运行 HTML 结构检查**

用现有脚本或一次性只读检查确认：

- 关键容器标签配平；
- 所有 `href="#..."` 都有唯一目标 id；
- 15 个 chapter anchor 各出现一次；
- 没有 `{{...}}` 残留；
- 模板、index、artifact 的关键 P1 契约都存在，旧架构语义都不存在。

**Step 2: 运行差异卫生检查**

Run:

```powershell
git diff --check
git status --short
```

Expected: 无 whitespace error；只保留本轮目标文件、已有用户修改和已知未跟踪文件，不出现意外文件。

**Step 3: 运行全仓测试**

Run:

```powershell
python -m pytest -q
```

Expected: 全仓测试通过；若存在环境无关 warning，记录数量但不掩盖失败。

**Step 4: 浏览器视觉检查**

启动本地只读 HTTP server，打开 `index.html`，至少检查：

- 首页标题和目录显示 `v4.1`；
- 架构、CodeIR/RuntimeEventPlan、Effect/Evidence、后端、18 门和 trace 验证章节；
- 代码块、表格、长中英文标识不溢出；
- 目录跳转和明暗主题切换可用；
- 删除旧图后不存在空 figure、断号或异常留白。

**Step 5: 最终复核**

再次运行：

```powershell
python tools/verify_gates.py
python -m unittest discover -s tools -p "test_*.py" -v
git diff --check
```

仅在这些命令和全仓测试均有新鲜成功证据后，报告修复完成；交付时列出修改文件、关键决策、测试结果和未实现的真实 conformance fixture（不得声称已完成真机验证）。
