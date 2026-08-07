# Target Design v2 Estimator Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 原位把 `target-design-v2` 从证明型 OOM/时间样本方案改成允许模型误差、可比较配置的内存与 step time 工程预测方案。

**Architecture:** 保留源码偏特化、统一 IR、注册域、逐层改写、未知算子阻断和两个只读后端。将横切数值契约统一改为 `Estimate<T> + EvidenceTag + Coverage`；内存后端重放确定的 `Allocate/Bind/Use/Free` 生命周期，时间后端按 OpDAG、stream、资源和 collective 关系执行 DES。为避免旧正文约 3800 行的证明支线继续互相引用，唯一正文源原位升为 v4 并重写成一份连贯规范，而不是对旧语义做字符串级补丁。

**Tech Stack:** HTML 模板、现有 CSS/构建脚本、Python 3 标准库核验器、PowerShell、浏览器本地 HTML 冒烟检查。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-08-07-target-design-v2-estimator-contract-design.md`。
- 不读取或修改 `cost_eval/` 实现代码；本任务只修改方案文档及其文档工具。
- `src/index.template.html` 是唯一正文源；`index.html` 和 `artifact.html` 只能由 `tools/build_doc.py` 生成。
- 保留统一 IR、注册表隔离、未知算子阻断、配置 DSL、Memory/Time 后端只读等已批准方向。
- 不输出容量/OOM verdict、reserved/碎片估计或形式证明；不建立 `SoundResult` 支线。
- 内存只按显式 storage lifecycle 计算 `peak_allocated_bytes`；alias/view 没有新 `Allocate` 就不重复计数。
- 时间输出标量 `predicted_step_time`；同模型、硬件和 metric 口径下允许配置比较，但结论始终标作预测。
- 真机 trace 是可选标定/误差评估输入，不是预测结果存在的前置条件。
- 任务开始前 `src/index.template.html`、`HANDOFF.md`、`tools/verify_gates.py` 已有用户未提交修改；不得提交这三个文件，以免把用户既有变更卷入 commit。
- 历史 `ADVERSARIAL-RECORD.md`、`DEFECT-LEDGER.md`、`r*-refute.md`、`DEFECT-INVENTORY.md` 不修改。

---

### Task 1: 把旧门数核验器改成 v4 产品契约核验器

**Files:**
- Modify: `docs/target-design-v2/tools/verify_gates.py`
- Test against: `docs/target-design-v2/src/index.template.html`

**Interfaces:**
- Consumes: UTF-8 模板正文。
- Produces: 退出码 0/1；验证 v4 标题、输出契约、生命周期契约、DES 契约、结构门集合和禁用旧术语。

- [ ] **Step 1: 记录旧核验器基线**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tools/verify_gates.py
```

Expected: 当前 v3 正文与旧门表一致，退出码 0。

- [ ] **Step 2: 将核验器改成 v4 契约检查，并先对旧正文运行**

核验器应定义以下常量：

```python
REQUIRED_TEXT = (
    "目标方案设计 v4",
    "工程预测仿真器",
    "MemoryEstimate",
    "StepTimeEstimate",
    "peak_allocated_bytes",
    "predicted_step_time",
    "Allocate",
    "Bind",
    "Free",
    "同口径配置可比较",
)

FORBIDDEN_TEXT = (
    "oom_verdict",
    "peak_interval",
    "peak_allocated.lo",
    "peak_allocated.hi",
    "SoundResult",
    "Sample[step_time]",
    "VERDICT-WITHDRAW",
    "可证下界",
    "配置间不可比较",
    "本工具不回答哪个配置更快",
)

EXPECTED_GATES = {
    "G-IR1", "G-IR2", "G-IR3",
    "G-MEM1", "G-MEM2", "G-MEM3", "G-MEM4", "G-MEM5", "G-MEM6",
    "G-TIME1", "G-TIME2", "G-TIME3", "G-TIME4", "G-TIME5", "G-TIME6",
    "G-REP1", "G-REP2", "G-REP3",
}
```

实现以下检查：

```python
def check_required(html: str, errors: list[str]) -> None:
    for text in REQUIRED_TEXT:
        if text not in html:
            errors.append(f"缺少 v4 必需契约：{text}")

def check_forbidden(html: str, errors: list[str]) -> None:
    for text in FORBIDDEN_TEXT:
        if text in html:
            errors.append(f"残留 v3 产品语义：{text}")

def parse_gates(html: str) -> list[str]:
    return re.findall(r'<tr data-gate="([A-Z0-9-]+)">', html)
```

并断言：gate ID 不重复、集合恰等于 `EXPECTED_GATES`；`id="c0"` 到 `id="c14"` 各出现一次；模板不得引用 `SVG:01-layering`、`SVG:02-provenance`、`SVG:04-structure-backward`，因为这三张图仍含旧证明语义。

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tools/verify_gates.py
```

Expected: FAIL，至少报告缺少 v4 标题/输出和残留旧语义。这是契约测试的红灯。

- [ ] **Step 3: 检查核验器自身质量**

Run:

```powershell
python -m py_compile tools/verify_gates.py
git diff --check -- tools/verify_gates.py
```

Expected: PASS。不要提交该文件；它在任务开始前已经 dirty。

---

### Task 2: 原位重写唯一正文源为 v4

**Files:**
- Replace in place: `docs/target-design-v2/src/index.template.html`
- Reuse: `docs/target-design-v2/src/style.css`
- Reuse compatible diagrams: `{{SVG:03-semantic-layer}}`, `{{SVG:05-identities}}`, `{{SVG:06-placement-execution}}`, `{{SVG:07-unknown-op-loop}}`

**Interfaces:**
- Consumes: 已批准设计中的 `Estimate<T>`、MemoryEventView 和时间 DES 契约。
- Produces: 带 `{{CSS}}`、`{{TOC}}` 占位符且可由现有构建器处理的单一 HTML 模板。

- [ ] **Step 1: 保存只读基线信息，不创建副本覆盖用户文件**

Run:

```powershell
git diff --numstat -- src/index.template.html
(Get-Content -Encoding utf8 src/index.template.html | Measure-Object -Line).Lines
```

Expected: 记录当前 dirty 状态和行数，用于最终确认没有误改其它文件。不要用 checkout/reset 恢复文件。

- [ ] **Step 2: 用一份连贯 v4 正文替换旧正文**

模板必须保持：

```html
<title>统一源码级 Model IR — 目标方案设计 v4</title>
{{CSS}}
<div class="wrap">
{{TOC}}
<main>
...
</main>
</div>
```

正文按以下 15 章组织，每章必须有唯一锚点 `c0`…`c14`：

1. `§0 产品契约与非目标`：允许误差的工程预测；不做 fit/OOM、reserved 或数学证明。
2. `§1 总体架构与分层`：S0 → CoreIR → TrainIR → PrecIR → ShardIR → RematIR → SchedIR → 两个只读投影。
3. `§2 Estimate、Evidence 与 Coverage`：给出设计规格中的完整结构；证据只解释，不改变类型。
4. `§3 事实源与配置正则形`：源码、注册域、配置、环境、标定；快照和 digest。
5. `§4 CoreIR 与源码偏特化`：policy-free 核、guard 处理、不可判 guard 阻断。
6. `§5 四层身份与 storage epoch`：TensorId/StorageInstanceId/PlacementId/ExecutionTargetId。
7. `§6 三注册域与未知算子闭环`：结构/内存/时间面隔离，缺当前后端必需面才阻断。
8. `§7 训练、反向与精度改写`：反向、累加、主副本与 dtype 都显式入图。
9. `§8 分片与分布式优化器`：TP/PP/DP/CP/EP、bucket、collective、prefetch/free 策略。
10. `§9 重算与 SchedIR 双投影`：确定的 MemoryEventView 与 OpDAG/ResourceRequest 时间投影。
11. `§10 两后端算法与输出`：内存事件重放公式、DES 公式、两个输出 schema、同口径比较。
12. `§11 阻断与诊断`：未知 op、缺 shape/bytes/duration、非法生命周期、非法 DAG；绝不零填。
13. `§12 结构门与可选精度评估`：18 个 `<tr data-gate="...">`，ID 与 Task 1 完全一致。
14. `§13 复杂度、配置 sweep 与迁移`：每个配置完整仿真；差分携带 coverage/assumption 变化。
15. `§14 决策记录与诚实边界`：alias、lifetime、stream、资源、时长均可能双向偏差；无调度 trace 仍可出预测。

内存章节必须逐字包含等价公式：

```text
Allocate(s, b) ⇒ live_allocated += b
Free(s)        ⇒ live_allocated -= bytes(s)
peak_allocated_bytes := max_event live_allocated(event)
```

时间章节必须逐字包含等价公式：

```text
start_time(op) := max(dependency_ready, stream_ready, resource_ready, collective_ready)
end_time(op)   := start_time(op) + predicted_duration(op)
predicted_step_time := max(end_time) - step_start_time
```

报告 schema 必须包含设计规格 §4.2 与 §5.2 的全部字段。配置比较必须写明三个前提：同一模型快照、同一硬件快照、同一 metric 定义。

- [ ] **Step 3: 运行契约核验器取得绿灯**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tools/verify_gates.py
```

Expected: PASS，打印必需契约、禁用术语和 18 个结构门均一致。

- [ ] **Step 4: 做模板级全文检查**

Run:

```powershell
rg -n 'oom_verdict|peak_interval|peak_allocated\.lo|peak_allocated\.hi|SoundResult|Sample\[step_time\]|VERDICT-WITHDRAW|可证下界|配置间不可比较|本工具不回答哪个配置更快' src/index.template.html
rg -n '\{\{SVG:(01-layering|02-provenance|04-structure-backward)\}\}' src/index.template.html
git diff --check -- src/index.template.html
```

Expected: 两条 `rg` 均无输出，`git diff --check` 通过。不要提交该文件；它在任务开始前已经 dirty。

---

### Task 3: 更新交接摘要为 v4 当前状态

**Files:**
- Replace in place: `docs/target-design-v2/HANDOFF.md`

**Interfaces:**
- Consumes: v4 正文及构建/核验命令。
- Produces: 零上下文接手者可以直接继续审查 v4 的短交接页；历史审查细节仍由未修改的历史文件承载。

- [ ] **Step 1: 写入新的交接结构**

`HANDOFF.md` 必须只含以下当前信息：

- 一句话目标与明确非目标；
- 唯一正文源和生成物关系；
- 保留的架构主线；
- MemoryEstimate/StepTimeEstimate 输出摘要；
- 内存 lifecycle 与时间 DES 关键公式；
- 18 个结构门的类别摘要；
- 构建、核验、标签/锚点和浏览器冒烟命令；
- dirty-worktree 注意事项；
- 历史文件是过程记录、正文优先的说明。

不得继续把旧的单侧下界、三值判断、时间样本或旧门数描述成当前状态。

- [ ] **Step 2: 检查交接页当前摘要**

Run:

```powershell
rg -n '目标方案设计 v4|MemoryEstimate|StepTimeEstimate|Allocate|predicted_step_time' HANDOFF.md
rg -n '当前.*oom_verdict|当前.*peak_interval|当前.*Sample\[step_time\]' HANDOFF.md
git diff --check -- HANDOFF.md
```

Expected: 第一条覆盖所有关键词，第二条无输出，diff check 通过。不要提交该文件；它在任务开始前已经 dirty。

---

### Task 4: 重建页面并执行机器一致性检查

**Files:**
- Generated: `docs/target-design-v2/index.html`
- Generated: `docs/target-design-v2/artifact.html`

**Interfaces:**
- Consumes: v4 模板、CSS 和兼容 SVG。
- Produces: 可独立打开的两个 HTML 页面。

- [ ] **Step 1: 构建生成物**

Run:

```powershell
python tools/build_doc.py
```

Expected: 构建成功，无未替换占位符或缺图错误。

- [ ] **Step 2: 对模板和生成物运行产品契约检查**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tools/verify_gates.py
rg -n 'oom_verdict|peak_interval|peak_allocated\.lo|peak_allocated\.hi|SoundResult|Sample\[step_time\]|VERDICT-WITHDRAW|可证下界|配置间不可比较|本工具不回答哪个配置更快' src/index.template.html index.html artifact.html
```

Expected: 核验器 PASS，`rg` 无输出。

- [ ] **Step 3: 检查标签配平与悬空锚点**

Run a PowerShell script that:

1. 从模板和 `index.html` 中剥离 `pre/script/style/svg`；
2. 对 `div, ul, ol, table, p, li, tr, td, th, dl, dt, dd` 比较开闭标签计数；
3. 比较 `href="#..."` 集合与 `id="..."` 集合；
4. 确认 `{{...}}` 只在模板中出现，且模板只剩 `CSS/TOC/SVG` 合法占位符；生成物中不得出现任何占位符。

Expected: 配平缺陷为空、悬空锚点为空、生成物占位符为空。

- [ ] **Step 4: 检查生成物来源一致**

Run:

```powershell
git diff --check -- src/index.template.html index.html artifact.html tools/verify_gates.py HANDOFF.md
git status --short -- src/index.template.html index.html artifact.html tools/verify_gates.py HANDOFF.md
```

Expected: diff check 通过；状态只显示预期文件。不要提交这些文件。

---

### Task 5: 浏览器视觉冒烟与最终语义审计

**Files:**
- Inspect: `docs/target-design-v2/index.html`
- Inspect: `docs/target-design-v2/artifact.html`

**Interfaces:**
- Consumes: 已构建 HTML。
- Produces: 对目录、表格、代码块、兼容 SVG、长行溢出和章节跳转的人工验收结论。

- [ ] **Step 1: 启动本地静态服务并打开页面**

Run:

```powershell
python -m http.server 8766 --directory docs/target-design-v2
```

用浏览器打开 `http://127.0.0.1:8766/index.html`。检查首页、§2、§9、§10、§12、§14。

- [ ] **Step 2: 视觉检查关键结构**

逐项确认：

- TOC 中 15 章均可跳转；
- `Estimate<T>`、MemoryEventView、DES 三个代码块没有截断；
- 18 行结构门表在桌面宽度下可横向滚动且不覆盖正文；
- `03/05/06/07` 四张 SVG 无旧证明术语、无裁切；
- `index.html` 与 `artifact.html` 的正文视觉一致；
- 页面没有把“预测”渲染成警告或失败状态。

- [ ] **Step 3: 最终源级审计**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tools/verify_gates.py
python tools/build_doc.py
git diff --check
git status --short
```

Expected: 两个脚本退出码 0，diff check 无错误。记录工作区中哪些文件在任务开始前已 dirty，哪些生成物由本任务更新；不得声称其它既有修改属于本任务。

- [ ] **Step 4: 交付**

最终回复必须列出：

- 新产品契约；
- 内存与时间算法的最终口径；
- 修改文件；
- 构建、核验、标签/锚点和浏览器检查结果；
- 未修改实现代码、未增加 OOM/reserved 输出、未采集调度级 trace 的范围说明。
