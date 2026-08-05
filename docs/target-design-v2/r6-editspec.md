# R6 编辑规格 · 档 2/档 3 未落地条目

> 依据：正文 `docs/target-design-v2/src/index.template.html`（2345 行全读）· `DEFECT-LEDGER.md` · `ADVERSARIAL-RECORD.md` §6.1/§6.1b/§6.1c。
> 本文**不改文件**，只给可直接落盘的编辑规格。锚点均取自正文原文，已核对唯一性。

---

# 1. 核实结论（台账"现状"列不可信，以下为逐条回正文核实的结果）

## 1.1 台账说"已处理"、正文核实确认（不再动）

| 条目 | 正文落点 |
|---|---|
| L-026 | §10.4 why 框（行 1687-1698）逐字写了"分子不含 `replication` 是有意的"及理由 ✓ 与 §6.1c 裁定一致 |
| L-027 | §3.3(b) 行 682 `sweep_dims` 已是类型级导出 ✓ |
| L-028 | §12.2 `G-X3` 行 1903：域片 := 凸包、`residual_rel` 由留一交叉验证算出 ✓ |
| L-029 | §11.3 行 1789-1799 射程已扩到五类事实源 ✓ —— **但引入新冲突，见 §2 的 L-031′** |
| L-032 | §10.4 行 1653 第四触发项 `Σ own_residual_width / peak.lo > θ_res` ✓ |
| L-033 | §2.6 honest 框行 551 `P_test` 下限机器导出 ✓ |
| L-034 | §8.1 行 1245/1248 枚举域含 `synth_placement_faces` ✓ |
| L-041 | §1.2 行 118-122 准入已加；`G-L1` 仍为后置条件 ✓ 与 §6.1 裁定 #5 一致 |
| L-043 | §12.1 行 1863-1864 `G-A*` ✓ |
| L-044 | §0.2 判据 4 行 79 已含方差一问 ✓ |
| L-046 | §0.1 行 54-65 falsification 挂区间 ✓ |
| L-051 | §2.6 行 536 hull 已含参数字节与 saved/captures 字节 ✓ |
| L-052 | §12.2 `G-T1` 行 1970-1974 必建模族正表 ✓（`G-N2` 已按 §6.1b 降为实例） |
| L-059 | §11.1 行 1740 两类阻断码待遇 ✓ |
| L-076 | §12.2 行 1995-1996 `G-N3`/`G-B4` 已划出门表 ✓ |

## 1.2 台账未列、但核实为**已落地或已被间接解决**（不需要新规格）

| 条目 | 判定 | 依据 |
|---|---|---|
| **L-047**（`G-B3a` 该不该恢复为门） | **已被间接解决** | 正文行 1942 已按 §6.1 裁定 #1 走"收窄断言"：加限定语「由 `Pass_shard` 合成的」并保留后置条件。R4-C B-6 的反例（源码显式写的 `all_reduce`）被正文逐字写进了限定语的理由里。**冲突 #1 已闭合，门数不变** |
| **L-019**（G0.5 对外原子 × PGRO） | **已被间接解决**（档 1 遗留） | 全文已无 `G0.5` 这一层：§1.3 层序为 G0→G1→G1.7→G2→G2.5→G3→G4，§7.1 明写三个前段 pass "不独立成层……三者与 `Pass_train_step` 合为 G1"。冲突的两个前提之一消失 |
| **L-072 前半**（`peak_lower` 未写死取 `.lo`） | **已落地** | §10.2 行 1432 `peak_allocated.lo := inf_interleave( size := size.lo )` 已钉死。**后半（覆盖率报表项）未落地**，见 §2 的 E5 |
| **L-031 之 (3)**（首页 `verdict_reachability_cost`） | **已被间接解决** | §10.3 第一段输出 `blocking_set_size`，§10.5 强制打印 `S_∞ ∩ live(t*)` 完整成员清单，二者合起来就是该数 |
| **L-049 之 (b)**（诚实声明无计数/无 `clean_effect`） | **已被间接解决** | L-029 落地后 §11.3 行 1797 明列"绑定条数"为 `clean` 向量的计数分量 |
| **L-058 之 (a)(c)**（`G-T4` 的 gather 计数 / 重算区子序列） | **已被间接解决** | `G-T1` 加必建模族正表后，集合通信族与已注册 op 族的**欠建模与过建模都硬失败** ⇒ 计数相等已被蕴含。**只剩"坍缩权等价类"半边未落地**，见 §2 的 E9 |
| **L-030 的承重论证** | **前提被消解** | R4-2 §3.5 的立论是"录制 trace 会被 `G-T0b` 判为不可用 ⇒ 5 道同时失 fixture"。但 `G-M1` 的 fixture 是**缺陷注入 fixture**，其 `EnvFacts` 与录制 trace 同属该 fixture、自洽，`G-T0b` 在 fixture 内不会红。叠加 E4（允许合成 op 序列作 fixture oracle）后前提完全消解。**只落地它的定性禁令**，见 §2 的 E12 |

## 1.3 判定为**未落地 / 部分落地**（本文出规格）

档 2：**L-031′（新冲突）· L-035 · L-037 · L-038 · L-039 · L-040 · L-042 · L-045 · L-048 · L-050 · L-054 · L-056 · L-057 · L-058(残) · L-030(残)**
档 3：**L-060 · L-061 · L-062 · L-063 · L-064 · L-065 · L-066 · L-067 · L-068 · L-070 · L-071 · L-072(残) · L-073 · L-074(a)**
另有三条**已落地条目的残留**（台账未单列，核实新发现）：**L-036 残 · L-053 残 · L-055 残**

## 1.4 判定为**不改**（理由见 §4）

**L-049(a)** · **L-069** · **L-075** · **L-074(b)** · **L-031 之 (2) 的枚举式形态** · **L-030 的 `ProbeSuite`/`G-T5` 构件**

---

# 2. 编辑规格（按优先级排序）

## 优先级 T1 —— 使某构件恒真 / 恒假 / 无生产者 / 两处规定不能同时为真

---

### E1 · L-031′ · `L-mono` 射程扩张后，"去标定"这个动作违反 `L-mono`，而 `G-M2b` 变绿的唯一路径正是它

现状核实： §11.3 行 1781-1783 定义 `L-mono` 为「对任意补救 r（向输入集增加一条声明）clean(I∪{r}) ⊑ clean(I) 且至少一个分量严格下降」；行 1796-1799 把量词扩张为「任何向输入集增加条目的动作，五类事实源全部在内……去 `CalibrationSet` 加一个标定点」。而 `clean` 的分量 `c3 = oom_verdict ≠ undetermined`（§11.3 行 1774）。加一个标定点 ⇒ 该量退出 P5 兜底 ⇒ `peak.hi` 向可计算前进（§2.5 P5 行 472-474 逐字如此写）⇒ `c3` 由 `false` 升为 `true` ⇒ **`clean` 不满足 `⊑`**。同一件事在 §10.2b 行 1593-1595 被写成 `G-M2b` 的验收路径：「标定工作一旦推进到外部碎片，`G-M2b` 会自己变绿」。**两条已落地的修复（L-029 的射程扩张 与 §10.2b 的 `G-M2b`）在规则上不能同时为真**：按 `L-mono`，使 `G-M2b` 变绿的那个动作是非法的；按 `G-M2b`，它是本方案唯一的验收条件。此外 `G-B0` 的 schema 位宽仍写 `Vector[5]`（行 1786），而 L-029 又要求每类事实源各加一个计数分量，位宽与内容对不上。

判定：    自拟修法。台账无此条（它是 L-029 落地的产物）。理由：不能靠"标定不算补救"这句话解决——L-029 的整个立论就是"自由度已从补救迁到事实源声明"。正确的切分不是**通道**（补救 vs 事实源）而是**条目的性质**：一条条目是不是**一次测量**。判据机器可判（其产物是否带 `calib` 根，或是否附 `EnvProbe` 的执行结果），不新增字段，且与 §2.5 P5「标定集是真正的交付物」、§3.1「标定是链的输入不是链的裁判」两处立论同向。

锚点（1/3）：
```
L-mono（不可逆脏化律）：对任意补救 r（向输入集增加一条声明）
    clean( Report(I ∪ {r}) )  ⊑  clean( Report(I) )     逐分量偏序
    **且至少一个分量严格下降**
```
替换为：
```
L-mono（不可逆脏化律）：射程按条目**是不是一次测量**二分。判据机器可判、无新字段：
    该条目的产物带 calib 根，或附 EnvProbe 的执行结果 ⇒ **测量类**；否则 ⇒ **声明类**

  **测量类**（CalibrationSet 的点、EnvProbe）：**不受 L-mono 约束**。
      它使区间收紧、允许 c1 与 c3 上升——这正是 §2.5 P5 与 §10.2b 写死的那条产品路线
      （标定一个量 ⇒ 它退出单侧 ⇒ peak.hi 向可计算前进一步 ⇒ G-M2b 自己变绿）。
      它的代价由"必须真的去测"承担，**不由 clean 向量承担**。
      **不这样切，L-mono 会禁掉 G-M2b 唯一的变绿路径**，两条规定不能同时为真。

  **声明类**（注册条目、placement 声明、分类表改判、PolicyStateBinding、域片划分、补救模板）：
      对任意声明类条目 r
          clean( Report(I ∪ {r}) )  ⊑  clean( Report(I) )     逐分量偏序
          **且至少一个分量严格下降**
```

锚点（2/3）：
```
<p><b>所以 <code>L-mono</code> 的量词是"任何向输入集增加条目的动作"</b>，
五类事实源全部在内。每类各自定义它进 <code>clean</code> 向量的计数分量
（标定点数、绑定条数、分类表改动行数、域片数、注册条目数），
新增即计入，且 <code>G-B0</code> 的 CI 实测对它们同样适用。</p></div>
```
替换为：
```
<p><b>所以 <code>L-mono</code> 的量词是"任何向输入集增加<b>声明类</b>条目的动作"</b>，
五类事实源里的<b>声明面</b>全部在内。每类各自定义它进 <code>clean</code> 向量的计数分量
（绑定条数、分类表改动行数、注册条目数、placement 声明条数、补救模板条数），
新增即计入，且 <code>G-B0</code> 的 CI 实测对它们同样适用。</p>
<div class="cal why" style="margin:14px 0 4px"><span class="lbl">为什么"标定点数"不在这份计数分量里</span>
<p>把标定点也算成脏化，会让 <b>唯一一条使答案变紧的诚实动作</b>受罚：加一个标定点使
<code>c3</code>（<code>oom_verdict ≠ undetermined</code>）可能由 <code>false</code> 升为 <code>true</code>，
逐分量偏序 <code>⊑</code> 当场不成立。而 §10.2b 把"<code>G-M2b</code> 会自己变绿"写成了标定路线图的
<b>验收条件</b>——两条规定不能同时为真。</p>
<p>切分不在<b>通道</b>上（补救 vs 事实源），而在<b>条目的性质</b>上：<b>测量</b>把不确定性变小并留下指纹，
<b>声明</b>把不确定性改名。判据 5 要罚的从来是后者。</p></div></div>
```

锚点（3/3）：
```
    每条补救模板在 schema 中必填 clean_effect : Vector[5] of {不变, 下降}，且至少一项为「下降」；
```
替换为：
```
    每条**声明类**条目在 schema 中必填
        clean_effect : Vector[5 质量分量 + 5 计数分量] of {不变, 下降}，且至少一项为「下降」；
        （位宽随 clean 向量定义走，**不得写死为 5**——L-mono 扩射程时已加了五个计数分量）
```

影响：  不改门表、不改门数。改首页字段 `clean` 的定义与位宽。**与 L-029（已落地）、§10.2b `G-M2b`、§2.5 P5 强耦合**——必须最先落，否则 E4（`c5`）与 E5（覆盖率）的判据 5 论证都建在一条自相矛盾的律上。

---

### E2 · L-039 + L-040 · `class_origin` 在新框架版本上恒为 `tool_shipped`，且它逐行不逐列

现状核实： §3.3 行 666-667 仍写 `PolicyClassTable.row : { field, class, pi0, reintroducer, class_origin }` / `class_origin : tool_shipped | user_override # 由与出厂表比对得出，**不可填**`。两个后果：(a) 框架出新版时 `factory_table[v_new]` 无定义，唯一可行路径是使用者为 `v_new` 生成出厂表 ⇒ 每行 `class_origin = tool_shipped` ⇒ §3.3(a)(b) 两条后果（`abstraction(policy_reclassification)` 根 + 退出扫描）**永不触发**；(b) `class_origin` 逐行不逐列 ⇒ 只改 `pi0` 或 `reintroducer` 而不改 `class`，两条后果都不触发、**代价为零**，而这两列是承重的（`pi0` 是裁决一的断言对象，`reintroducer` 既是 PGRO 的形式参数又是 §3.5 `G-E8` 授权集的导出源）。§3.2 行 656-658 已为 `Π_semantic` 写了"PE 分叉机器产出 + 与出厂表比对，不一致 ⇒ 阻断"，但**只对一类**。

判定：    采纳台账 L-039（R3-1 A10-a）与 L-040（R3-1 A10-b），合并为一处编辑。理由：两条改的是同一个 `pre` 块，且 A10-a 提供的机器导出正是 A10-b 逐列比对所需要的基线——没有机器导出，逐列 `class_origin` 在新版本上同样恒为 `tool_shipped`。`G-P3`（已落地）只保证两处 π₀ 一致，不保证它们与出厂/机器基线一致，所以它接不住 (b)。

锚点：
```
<pre><code>PolicyClassTable.row : { field, class, pi0, reintroducer, class_origin }
class_origin : tool_shipped | user_override      # 由与出厂表比对得出，**不可填**</code></pre>
```
替换为：
```
<pre><code>PolicyClassTable.row : { field, class, pi0, reintroducer,
                         origin : { class: O, pi0: O, reintroducer: O } }   # **逐列，不是逐行**
O := tool_shipped | user_override                # **不可填**，由与基线比对得出

基线**不是出厂表**，是 PE 的机器导出——否则框架出新版时 factory_table[v_new] 无定义，
使用者只能自己生成一张，其中每行都是 tool_shipped ⇒ §3.3 的两条后果**永不触发**：

  机器导出的 class（把 §3.2 对 Π_semantic 的做法推广到全部类）：
    PE 在该字段上分叉，按**客观判据**归类——
      两支的 op 多重集不同，且差集含参与梯度/损失的节点  ⇒ Π_semantic
      两支的 op 多重集不同，差集不参与梯度/损失          ⇒ Π_impl
      两支的 op 多重集相同，只有广延量不同                ⇒ Π_extent
      两支残差同构                                       ⇒ Model / Env
  机器导出的 pi0：该字段的单位元由 §3.2 的定义给出（并行度 1、重算 None、不融合、全精度、无流水）
  机器导出的 reintroducer：唯一一个 own_roots 含该字段的 pass（P4 表逐 pass 可查）

  user_override(列 x) :⟺ 声明的 x ≠ 机器导出的 x       # 有无出厂行都可判
</code></pre>
<p><b>逐列而不逐行</b>：`class_origin` 逐行时，只改 <code>pi0</code> 或 <code>reintroducer</code>
而不动 <code>class</code>，§3.3 的两条后果都不触发、<b>代价为零</b>——而这两列是承重的
（<code>pi0</code> 是裁决一的断言对象本身，<code>reintroducer</code> 同时是 PGRO 的形式参数与
<code>G-E8</code> 授权集的导出源）。逐列之后：<b>改 <code>reintroducer</code></b> ⇒ 该字段的 PGRO 结论携带
<code>abstraction(policy_reclassification)</code> 根；<b>改 <code>pi0</code></b> ⇒ π₀ 全量进
<code>model_identity_digest</code>，两次运行不再可比。</p>
```

影响：  不改门表；`G-P3`（已落地）与本条互补（它管两表一致，本条管与机器基线一致）。改首页 `excluded_from_sweep` 的产生条件。与 E10（`Π_impl` 的 π₀ 准入义务）同在 §3.2/§3.3，先后无关但同一批。

---

### E3 · L-035 · `G-B4` 的"取值均匀"是一个不在 Θ 清单里的隐藏阈值

现状核实： §2.8 行 607-611 `G-B4` 仍写「一组量满足算子声明的守恒约束……<b>且取值均匀</b>」，理由列写「均匀分布 + 守恒是"完美均衡"的<b>充要指纹</b>」。L-076 已把 `G-B4` 移出门表、移入 §2.8 的代数 ✓，但"均匀"这个词原样保留。取严格义 ⇒ 写 `[1024,…,1023,1025]`（守恒仍成立）即不触发 ⇒ 落回 `configured/exact` ⇒ §2.8 三道闭合被绕过；取近似义 ⇒ 它是一个可调宽窄的阈值，而 Θ 是闭集、§11.5 行 1843 明禁"使用者可设的门限"。

判定：    采纳台账修法（R3-1 A13）。理由：它**消灭一个阈值且判据更强**——正是 §12.2 `G-X3` 那一段刚刚用过的招（"凡'声明一个集合'的地方，都该问它能不能改成'由已有事实导出'"）。反向的代价是零：守恒约束由算子在 S1 声明，是闭集里已有的东西。

锚点：
```
<td>一组量满足算子声明的守恒约束（如 <code>Σ_experts tokens_e = topk · num_tokens</code>）<b>且取值均匀</b>
```
替换为：
```
<td><b>无参数的结构判据</b>：在 <code>axis_kind = data_dependent</code> 的位置上，<b>一切</b>满足算子声明的
      守恒约束（如 <code>Σ_experts tokens_e = topk · num_tokens</code>）的取值组，<b>不论是否均匀</b>
```

锚点（同一 `<tr>` 的理由列）：
```
  <td>均匀分布 + 守恒是"完美均衡"的<b>充要指纹</b>，机器可识别，且它是一个假设而不是一个配置选择</td></tr>
```
替换为：
```
  <td><b>守恒约束成立本身就说明这组数是被"配平"出来的，不是被观测出来的</b>——它是一个假设，
      不是一个配置选择。<b>判据里不能出现"均匀"</b>：取严格义则写
      <code>[1024,…,1023,1025]</code>（守恒仍成立）即可绕过，三道闭合全部落空；
      取近似义则"多均匀算均匀"是一个阈值，而 Θ 是闭集、§11.5 禁使用者可设的门限。
      去掉这个词，判据既无参数又<b>严格更强</b></td></tr>
```

影响：  不改门表（`G-B4` 已在 §2.8 内）、不改门数。改 §2.8 三道闭合的射程（变宽）。与 E14（档 A/B′ 划分）弱耦合：`G-B4` 判出的量其区间仍来自来源④，落档 B′。

---

### E4 · L-038 · `c5` 在 `G-B0` 的 CI 里恒 0 ⇒ `L-mono` 实际只有三个可用分量

现状核实： §11.3 行 1778 `c5 = 被独立对照门覆盖的算子占比`；行 1787-1788 `G-B0` 用**合成 fixture** 实跑两次。而 §12.3 行 2030 已把"相互独立的外部门"收敛到 **只有 `G-T1` 一道**，`G-T1` 的 oracle 是真机 op trace。合成 fixture 没有 trace ⇒ `G-T1` 无 oracle ⇒ 两次都是 0 ⇒ `c5` **恒 0**、恒"不变" ⇒ 任何以 `c5` 为唯一下降分量的补救模板必被 `G-B0` 拒绝。叠加 §6.1b 的降级（`G-N2` 不再独立计数），`c5` 的定义本身也已失去所指。

判定：    采纳台账修法的**第一支**（R3-3 §2.8 的"允许合成 fixture 附带一条合成 op 序列作为 oracle"），不采纳第二支（"`c5` 只在带真机 trace 的集成测试里参与、不进 `G-B0`"）。理由：第二支把 `L-mono` 的可用分量从 5 减到 4，而 `c1`（`exact` 字节占比）在显存链上本就近乎不动、`c3` 已按 E1 划归测量类不受约束 ⇒ 实际只剩 `c2`、`c4` 两个分量承担全部代价单调性，判据 5 的载体过窄。第一支不违反 P-oracle（合成序列不是实测数据、无指纹，`G-X4` 不适用），且合成序列由 **fixture 作者**写、不由用户写，不构成新的逃逸通道。

锚点：
```
  c5 = 被独立对照门覆盖的算子占比
```
替换为：
```
  c5 = 被 G-T1 匹配上的 op 占比
       # 不写"被独立对照门覆盖"：按 §12.3，相互独立的外部门**只剩 G-T1 一道**，
       #   旧措辞的所指已随 G-N2 降为实例而消失
```

锚点：
```
    断言实测方向与声明一致。合成 fixture 不需要框架源码。
```
替换为：
```
    断言实测方向与声明一致。合成 fixture 不需要框架源码。
    **合成 fixture 必须附带一条合成 op 序列作为 G-T1 的 oracle**，否则 c5 在 CI 内恒 0、
    恒"不变"，以 c5 为唯一下降分量的补救模板必被本门拒绝，L-mono 退化为只由 c2 与 c4 承担。
    合成序列**不是实测数据**（无指纹、不进 CalibrationSet）⇒ 不触 P-oracle、不触 G-X4；
    它由 fixture 作者写而不由使用者写 ⇒ 不是新的逃逸通道。合成序列进 fixture 的 digest。
```

影响：  不改门表、不改门数。改首页 `clean` 的 `c5` 定义。**与 E1 强耦合**（E1 把 `c3` 划出 `L-mono` 之后，`c5` 的可用性成为判据 5 的必要条件）。也消解 L-030 的承重论证（见 §4）。

---

### E5 · L-054 + L-072(残) · `peak_allocated.lo` 不是可证下界，而它是显存链唯一可被真机 OOM 证伪的构件

现状核实： §10.2 行 1469-1474 已按 L-008 给 alias 走来源④，写死 `lo 端 = 按我们的等价类求和（最粗划分：判为 alias 的合并成一块）`、`hi 端 = 每个 TensorId 各占一块`。**但它只是"我们的判定"**：若某两个实际共享存储的值被我们判成 `new`，最粗划分也不会把它们合并 ⇒ 求和偏高 ⇒ `lo` 端**不是**下界。行 1475-1479 的"条件一"原样承认了这一点（"可能产出假的 OOM 断言……它是一个条件下界，不是一个定理"），即 L-054 的"未收紧"仍在。同时 §0.1 行 62-65（L-046 已落地）把 falsification 挂在了 `peak.hi < C_eff.lo` 与 `peak_allocated.lo` 上 ⇒ **一个非下界的量正在承担全案唯一的外部反证**。L-072 后半（`peak_lower_covered_bytes / peak_lower` 报表项）亦未落地。

判定：    采纳 §6.1 裁定 #9 已选的那一支（R4-1 R5「合并」），并把它从"取我们的等价类"收紧为"**取证据加权的最粗划分**"。理由：裁定 #9 的理由（合并后求和仍精确 ⇒ 不吃 `own_residual` ⇒ 不落兜底 ⇒ `true` 在类型上可达）只有在**合并规则本身不依赖我们可能判错的那一步**时才成立。R3-1 A5 的"丢弃"（非 `exact` 的一律按 0 计）被否，因为它会把持久存储（正是 §10.2 行 1488 论证"排除之后还剩多少"的主力）整块删掉，下界退化为平凡。

锚点：
```
<p><b>条件一 · alias 等价类必须正确。</b>若两个实际共享存储的值被判成两个 <code>StorageId</code>，
```
替换为：
```
<div class="cal inv" style="margin:14px 0"><span class="lbl">条件一 · <code>lo</code> 侧的划分必须按<b>证据</b>取最粗，而不是按<b>我们的判定</b>取最粗</span>
<p>"最粗划分 = 判为 alias 的合并成一块"<b>还不够</b>：错在另一个方向的那一半没有被吃掉——
若两个实际共享存储的值被我们判成两个 <code>new</code>，最粗划分也不会合并它们
⇒ 各计一次 ⇒ 求和偏高 ⇒ <code>lo</code> 端<b>不是</b>下界，而 §0.1 的 falsification 与
§10.2b 的"会 OOM"都建在它上面。</p>
<pre><code>peak_allocated.lo 的求和划分 := **证据加权的最粗划分**
  凡满足「判为 new，但产出它的算子其 S1 storage 面 evidence.grade ≠ calibrated
        且 origin.readability ≠ FULL」者              ← 即"非别名性没有源码事实支撑"
  与其输入**合并进同一 StorageId** 再求和
  合并只会**降低**求和 ⇒ 在 alias 规则的**任何**错误下仍是下界 ⇒ certainty 可为 exact
  ⇒ 不吃 own_residual ⇒ **不落 P5 兜底** ⇒ verdict 的 true 侧在类型上可达</code></pre>
<p><b>为什么不是"丢弃"</b>（把证据不足的 storage 一律按 0 计）：那会把<b>持久存储</b>整块删掉，
而它正是本节下面论证"排除之后还剩多少"的主力（16 B/param）⇒ 下界退化为平凡，
"<code>inf_interleave ≥ Σ 参数字节</code>"这条可证伪断言随之消失。<b>合并更紧且同样安全。</b></p>
<p>合并集的基数 <code>|merged_by_weak_evidence|</code> 与
<code>peak_allocated_lo_covered_bytes / peak_allocated.lo</code>（有多少下界是真有依据的）
<b>均为首页强制字段</b>——空洞必须可见，否则"条件下界"这句标注不产生任何后果。</p></div>
<p><b>条件一的残余风险仍须写明。</b>若两个实际共享存储的值被判成两个 <code>StorageId</code>，
```

影响：  不改门表、不改门数。**改首页字段**（新增两个）。**与 L-008（已落地）同段、与 L-046（已落地）强耦合**——L-046 的 falsification 判据依赖本条成立。落盘时本条必须先于 E13（首页"非界"行）。

---

### E6 · L-042 · `span(root)` 可被一条合法枚举值主动驱动出扫描，且在 `S_∞` 成员上无定义

现状核实： §13.3 行 2144 `span(root) = Σ_{s ∈ live(t*), root ∈ s.roots} (s.hi − s.lo)`。两处：(a) `S_∞` 成员至少一端不可计算 ⇒ `s.hi − s.lo` 无值 ⇒ 这些根在 top-N 排序里无位置，而它们恰是最该被列出的；(b) 求和域取 `live(t*)` ⇒ 声明 `grad_reduce.free_policy: after_reduce`（合法枚举值，`G-N1` 通过）即把分桶复刻规则覆盖的字节移出 `live(t*)` ⇒ 一个**未经验证的模型判定**把自己移出了敏感性扫描。§10.3 已把 `S_∞` 单列成第一段 ✓，但 §13.3 的 `span` 未同步。

判定：    采纳台账修法（R3-1 A4 的两条全取）。理由：(1) 与 §10.3 已落地的两段拆分是**同一招在另一处的补齐**——那里已经承认"宽度不可计算 ⇒ 无全序"；(2) 求和域从 `live(t*)` 扩到全部 storage 严格变大、方向保守，成本为零（同一遍 liveness 内即可算）。

锚点：
```
<code>span(root) = Σ_{s ∈ live(t*), root ∈ s.roots} (s.hi − s.lo)</code>
```
替换为：
```
<code>span(root) = Σ_{s ∈ **全部 storage**, root ∈ s.roots} (s.hi − s.lo)</code>
——<b>求和域不取 <code>live(t*)</code></b>：取了的话，声明 <code>grad_reduce.free_policy: after_reduce</code>
（合法枚举值，<code>G-N1</code> 通过）就把分桶复刻规则覆盖的字节移出了求和域
⇒ <b>一个未经验证的模型判定把自己移出了敏感性扫描</b>。扩到全部 storage 严格变大、方向保守，
且在同一遍 liveness 内即可算出，增量仍为零。
<b>而 <code>own_residual</code> 无机器来源的根（即 <code>S_∞</code> 的根）不用 <code>span</code> 排序</b>
——它们的端点不可计算、<code>s.hi − s.lo</code> 无值 ⇒ 无全序。这类根改用
<code>coverage(root)</code> 排序，并<b>单独占一类 top-N 名额</b>（档 C 本来就是分类截断，加一类成本为零）
```

影响：  不改门表、不改门数。改 §13.3 的 top-N 截断口径与 §10.3 第一段的排序键（二者本就"是同一张表"）。与 E14（档 B′）同节。

---

### E7 · L-070 · `Sample[T]` 无 `roots` 字段，而档 C 把"makespan 的 duration 根"当扫描维度

现状核实： §2.3 行 269 `Sample[T] := 观测样本的极值集合  # **不是界**；无区间算术、无比较、不进 OOM 链`——**无 `roots` 字段**。§13.3 行 2138 档 C 列「allocator 碎片 / 保留、`timing_dependent` 通信时长、**makespan 的 duration 根**」。第三项在类型上没有生产者。

判定：    采纳台账并列修法中的**第二条**（"档 C 的该维度改由 §9.2 的 `CalibKey` 族导出"），不给 `Sample[T]` 加 `roots`。理由：§9.2 行 1382-1383 **已经**把敏感性扫描的维度集合定义为"每条 `timing_dependent` 边所依赖的通信算子的 `CalibKey` 族"，而 L-011 已给 `CalibKey` 加了 `measured_quantity`（§3.1 行 626）——所需的两件东西都已在正文里，**零新构件**。反之给 `Sample[T]` 加字段会重新打开它的类型纪律：`G-X1`（行 1817）断言不存在 `Sample[T] → Q[T]` 转换算子，而一个带 `roots` 的 `Sample` 在数据上与 `Q` 只差 `interval`，正是复活检测最难看住的形态。

锚点：
```
<tr><td class="nw">C</td><td><b>非单调</b>维度：allocator 碎片 / 保留、<code>timing_dependent</code> 通信时长、makespan 的 duration 根</td>
```
替换为：
```
<tr><td class="nw">C</td><td><b>非单调</b>维度：allocator 碎片 / 保留、<code>timing_dependent</code> 通信时长、
  <b>makespan 所依赖的 <code>CalibKey</code> 族</b>（<code>measured_quantity = duration</code>，按 §9.2 导出）
  ——<b>不写"makespan 的 duration 根"</b>：<code>Sample[T]</code> 在类型上<b>没有 <code>roots</code> 字段</b>，
  那个维度会没有生产者。给 <code>Sample</code> 补 <code>roots</code> 是错的走法：带 <code>roots</code> 的
  <code>Sample</code> 与 <code>Q</code> 只差一个 <code>interval</code>，正是 <code>G-X1</code> 最难看住的复活形态</td>
```

影响：  不改门表、不改门数。与 L-011（已落地的 `measured_quantity`）、§9.2 的三个消费者（见 E19）耦合。

---

### E8 · L-071 · §12.3 的两个数当场就是错的（首页强制字段自己算错）

现状核实： §12.3 行 2010 写「真门 42；扣独立性折扣 2 ⇒ **独立真门 40**，按 oracle 分布」，其后的分布表逐行相加 = **42**，不是 40——标签挂错了对象。行 2039 写「这一刀切在全部 42 道真门上……：**17 : 28**」，而行 2031/2032 的两行表给的是 **16 : 26**（16 + 26 = 42 ✓，17 + 28 = 45 ✗）；同段"用自己写的对照物检模型事实的是 13 道"正是 16 − 3，说明 16 才是对的、17 是旧数残留。机制侧：行 2010 已写 `tools/verify_gates.py` 机器核对 ✓、行 2041 已写"由门表机器生成 + 断言各切分之和 == 真门数" ✓；**缺 `G-M1` 的那条子条款**（断言形态变更 ⇒ N 必须重算）。

判定：    采纳台账修法（R4-2 §6.4④）并**先修两个数**。理由：这两个数是 §10.5 强制上首页的字段，而 §12.3 自己写着"手数已经错过两次"——第三次就在同一段里。

锚点（1/3）：
```
⇒ <b>独立真门 40</b>，按 oracle 分布（<b>本表由 <code>tools/verify_gates.py</code> 机器核对</b>）：</li>
```
替换为：
```
⇒ <b>独立真门 40</b>。<b>下表按 oracle 分布的是 42 道真门，不是 40 道</b>——独立性折扣只作用于
"相互独立"这个切分，不改变每道门的 oracle 归属，两个数不能挂在同一张表上
（<b>本表由 <code>tools/verify_gates.py</code> 机器生成并核对</b>）：</li>
```

锚点（2/3）：
```
：<b>17 : 28</b>。而这 17 道里有 3 道
```
替换为：
```
：<b>16 : 26</b>。而这 16 道里有 3 道
```

锚点（3/3，加 `G-M1` 子条款）：
```
<b>所以编号规范扩一档</b>：新增前缀 <code>G-A*</code>（层不变量与散在断言），
逐条编号后与门表同列，<code>G-M1</code> 的量词随之覆盖它们。零设计成本，只是把已有的断言登记进来。</p></div>
```
替换为：
```
<b>所以编号规范扩一档</b>：新增前缀 <code>G-A*</code>（层不变量与散在断言），
逐条编号后与门表同列，<code>G-M1</code> 的量词随之覆盖它们。零设计成本，只是把已有的断言登记进来。</p>
<p><b><code>G-M1</code> 子条款 · 数必须跟着断言走</b>：任何一行的<b>断言形态</b>发生变更
（门 ⇄ 后置条件 ⇄ 报告项、独立性折扣的增减、oracle 归属改变），
"N 道独立门"与 §12.3 的三个切分<b>必须由门表重算</b>，<b>禁止手写</b>，
并断言"各切分之和 == 真门数"。<b>手数已经错过三次</b>，而它们是 §10.5 强制上首页的字段
——一个自己算错的首页数字，正是本门要消灭的那种虚假安全感。</p></div>
```

影响：  **改门数呈现（不改门的集合）**。落盘顺序上必须**排在 E11（新增 `G-C3b`）之后**，否则要改两次数。

---

### E9 · L-058(残) + L-074(b) · 坍缩权无规则：`G-T1` 一通过就把区间坍缩为点，是判据 5 的反向失败

现状核实： §8.3 行 1336 只有一条坍缩规则：`坍缩条件： 相关集合通信在 op 序列门中逐条匹配、时长残差在域内 ⇒ degree 获佐证 ⇒ 区间坍缩为点`。全文再无第二条关于"允许坍缩哪些区间"的规定。而 §8.2 行 1312-1316 已明写 `grad_reduce.free_policy` 的 `at_step_end` 与 `never` 在 `O` 中**结构性不可分辨**——若有人照着 §8.3 的形态把"trace 匹配上了"推广成"该区间可坍缩"，它会坍掉一个**在原理上无法被这条 trace 区分**的区间。台账 L-074(b) 提出的"`G-T1` 通过 ⇒ 峰值可在观测到的线性化上直接算"正是这条推广的最强形式。

判定：    采纳 L-058 的**坍缩权硬约束半边**（R4-2 §2.8 的后半），**不新增 `G-T4`**（其 (a)(c) 已被 `G-T1` 正表蕴含，见 §1.2）；**否决 L-074(b)**（理由见 §4）。理由：门数不变，改的是一条既有规则的边界；而这条边界正是判据 5 反向失败最直接的入口——"匹配上 trace ⇒ 答案更好看"会把 §14.2 已披露的"op 身份映射表被调宽则最强的门静默退化为恒真门"变成一个**有收益**的动作。

锚点：
```
坍缩条件： 相关集合通信在 op 序列门中逐条匹配、时长残差在域内 ⇒ degree 获佐证 ⇒ 区间坍缩为点
```
替换为：
```
坍缩条件： 相关集合通信在 op 序列门中逐条匹配、时长残差在域内 ⇒ degree 获佐证 ⇒ 区间坍缩为点

**坍缩权是闭集，不是"门通过就可以坍"**：
  只允许沿「**补偿 op 存在性等价类**」收窄——即两个取值在 O 中会留下**不同的**设备算子的那一对。
  **类内禁止坍缩**：grad_reduce.free_policy 的 {at_step_end, never} 在 O 中不可分辨（§8.2），
    其区间**必须原样保留**，哪怕 G-T1 全绿。
  等价类划分表进快照、参与 digest，并与"明示不建模族"同为闭集。

  **不设这条会怎样**：G-T1 一绿就允许把峰值改算在"观测到的那条线性化"上 ⇒ 区间坍缩为点
  ⇒ 匹配上 trace 使答案**更好看** ⇒ 判据 5 反向失败；而使 G-T1 更容易匹配的办法
  正是调宽 op 身份映射表——§14.2 已披露那会让最强的门**静默退化为恒真门**。
  ⇒ 把两件事接起来，就得到一条**有收益的**逃逸路径。
```

影响：  不改门表、不改门数。**新增一张进 digest 的闭集表**（补偿 op 存在性等价类）。与 §8.2（已落地）、§14.2 门体系侧、E12（op 身份映射表计数）耦合。

---

### E10 · L-050 · `Π_impl` 的 PGRO 只有 absorb 方向，展开方向静默失效

现状核实： §4.2 行 822 `G-absorbable` 的放行义务只有"注册一条逻辑原子 Op 后两支塌成同一节点"。§3.2 行 649 `Π_impl` 行只写"π₀ 取'逻辑原子形态'；由 `Pass_impl_select` 替换"，**没有规定 π₀ 该取哪个具体取值**。若框架该版本不提供 ungrouped 路径，π₀ 被迫落在**更融合**的一侧 ⇒ LHS 要求把一个 `readability = NONE` 的原子**展开**成 256 个 GEMM ⇒ 无任何义务、无任何 `BLK-*` 码覆盖"展开"方向 ⇒ PGRO 既不通过也不失败。台账所说的 "G0.5" 层已不存在（见 §1.2），修法须改述到 `Pass_impl_select`。

判定：    采纳台账修法（R3-3 §1.1），改述到 §3.2 的 `Π_impl` 行（而不是 §4.2 的 guard 表），因为它是一条**对 π₀ 取值的准入义务**，与 §3.6 同类。理由：放在 guard 表里会与 E17 的 `PG-*` 改名冲突；放在 §3.2 则同时钉死"π₀ 取逻辑原子形态"这句话现在是二义的（哪一个原子形态）。

锚点：
```
<td>π₀ 取"逻辑原子形态"；由 <code>Pass_impl_select</code> 替换。<b>替身判错的可见性由 <code>abstraction</code> 根承担</b></td></tr>
```
替换为：
```
<td>π₀ 取"逻辑原子形态"；由 <code>Pass_impl_select</code> 替换。<b>替身判错的可见性由 <code>abstraction</code> 根承担</b>。
      <b>准入义务（否则 PGRO 在这一类上静默失效）</b>：π₀ 必须取使 <code>Pass_impl_select</code>
      <b>输入侧 op 集合基数最大</b>的那个合法取值——即<b>最展开</b>的形态。理由：PGRO 的方向是
      "π₀ 残差 + 策略 ⇒ 重建"，pass 只会<b>合并</b>不会<b>展开</b>；π₀ 若落在更融合的一侧，
      LHS 要把一个 <code>readability = NONE</code> 的原子拆成 256 个 GEMM，而<b>没有任何义务、
      没有任何 <code>BLK-*</code> 码覆盖"展开"方向</b> ⇒ PGRO 既不通过也不失败。
      若最展开取值在该 <code>framework_ver</code> 下不合法（<code>G-N1</code> 拒），则该字段的 PGRO 义务
      改为<b>双向</b>（<code>π₀→v</code> 与 <code>v→π₀</code> 各判一次），任一方向失败即
      <code>BLK-UNRECONSTRUCTIBLE</code></td></tr>
```

影响：  不改门表、不改门数。改 §3.6 的准入义务集合（多一条）。与 E2 同在 §3.2/§3.3。

---

### E11 · L-048 · 无门覆盖"pass 现场调 PE 时少传一个字段"

现状核实： §3.5 行 757 已落地三道门 `G-E8`（授权闭包，管**越权**）、`G-E7a`（幂等）、`G-E7b`（无副作用）。**没有一道管"少传"**：`G-E8` 断言 `used_fields ⊆ 授权集`，少传只会让 `used_fields` 更小 ⇒ 恒过。`G-C3` 只跑 `P ∈ P_test`，而这类错误往往只在特定字段组合下非平凡（`padded_vocab` 只在 `vocab % (128·tp) ≠ 0` 时非平凡）。§12.2 门表无 `G-C3b`。

判定：    采纳台账修法（R4-1 R7(a)）。理由：正文删 `G-X6` 的二难（"重跑同一套抽取则恒真、换一套推导则不完备、相位错"）**对它不适用**——两个绑定环境是不同的对象、用的是同一个 PE、两个制品都在编译期存在。这是本轮唯一新增的门。

锚点：
```
<tr><td class="nw"><code>G-E8</code></td><td><b>授权闭包</b>：<code>used_fields ⊆ 该 pass 声明的授权字段集</code></td><td>pass 经规范化函数拿到未授权的策略字段（规范化函数是跨字段的）⇒ 层表签名不再为真</td><td class="nw">门</td></tr>
```
替换为：
```
<tr><td class="nw"><code>G-E8</code></td><td><b>授权闭包</b>：<code>used_fields ⊆ 该 pass 声明的授权字段集</code></td><td>pass 经规范化函数拿到未授权的策略字段（规范化函数是跨字段的）⇒ 层表签名不再为真</td><td class="nw">门</td></tr>
<tr><td class="nw"><code>G-C3b</code></td><td><b>现场求值一致性</b>：pass 每次现场调 PE 时记录 <code>(入口 anchor, 绑定环境规范形 digest, 输出 digest)</code>；断言存在一条 <code>PE(Src,·,P)</code> 的求值轨迹，其在<b>同一 anchor</b> 处的绑定环境 digest 相等</td><td><b>pass 调 PE 时少传一个字段</b>。<code>G-E8</code> 对它<b>恒过</b>（少传只让 <code>used_fields</code> 更小），<code>G-C3</code> 只跑 <code>P ∈ P_test</code> 而这类错误常只在特定字段组合下非平凡（<code>padded_vocab</code> 只在 <code>vocab % (128·tp) ≠ 0</code> 时非平凡）。<b>它不恒真</b>（两个绑定环境是不同对象）、<b>不需要第二套推导</b>（用同一个 PE）、<b>相位正确</b>（两个制品都在编译期）</td><td class="nw">门</td></tr>
```

影响：  **改门表、改门数**。落盘后 §12.3 全部数字变为：条目 **49**、后置条件 5、报告项 1、**真门 43**、折扣 2 ⇒ **独立真门 41**；oracle 分布"图自身或框架源码文本" **16 → 17**（成员追加 `G-C3b`），其余不变，合计 43；"检模型事实 : 自律门" = **16 : 27**（`G-C3b` 属"求值纪律"，归自律门，与 `G-E7`/`G-E8`/`G-P3` 同类）；"oracle 在工具之外 5，其余 **38**"。**必须先于 E8 落盘**，E8 的三个数按此写。

---

### E12 · L-057 + L-030(残) · op 身份映射表无计数；探针的定性禁令未进正文

现状核实： §14.2 门体系侧（行 2246-2248）已披露"op 身份映射表本身是又一张人工映射表，它若被调宽，最强的那道门会静默退化为恒真门" ✓；§2.3 行 254 `kind` 闭集已含 `op_identity_mapping` ✓。**计数未落地**：§10.5 首页表无相应字段。探针方面：全文无任何探针制度，也无 ADVERSARIAL-RECORD §3.3 要求"原样写进正文防止有人拿它复活该路径"的那条定性。

判定：    L-057 采纳台账修法（R4-2 §2.7），但把"覆盖的字节占比"改为**依赖该表的门的编号清单 + 道数**——理由：`G-T1`/`G-T2`/`G-N2` 全是时间侧的 op 序列门，"覆盖字节"对它们无定义；而按 §12.3 独立外部裁判只剩 `G-T1` 一道且它 100% 依赖该表，"1 道 / 100%"本身就是最有信息量的数。L-030 只采纳定性禁令（构件不落地，理由见 §4）。

锚点（L-057，插在首页表的 Θ 行之前）：
```
<tr><td class="nw"><code>Θ 清单 + 工具 digest</code></td><td>当次使用的全部阈值取值</td><td>—</td></tr>
```
替换为：
```
<tr><td class="nw"><code>op_identity_map_load</code></td><td><b>依赖 op 身份映射表的门</b>：编号清单 + 道数 + 该表的条目数 + 其中带 <code>abstraction(op_identity_mapping)</code> 根的条目占比。<b>当前取值是 <code>{G-T1, G-T2, G-N2}</code> / 3 道</b>，而按 §12.3 相互独立的外部裁判<b>只有 <code>G-T1</code> 一道 ⇒ 它对该表的依赖是 100%</b>。§14.2 已披露"该表被调宽则最强的那道门静默退化为恒真门"，本字段是把这句披露变成一个<b>可逐版本比较的数</b></td><td>—</td></tr>
<tr><td class="nw"><code>Θ 清单 + 工具 digest</code></td><td>当次使用的全部阈值取值</td><td>—</td></tr>
```

锚点（L-030 定性，接在 §8.3 honest 框末）：
```
已被前提 P-peak 关闭。</p></div>
```
替换为：
```
已被前提 P-peak 关闭。</p>
<p><b>另一条看起来能救它的路，必须在这里堵死</b>：主动造 oracle 的<b>判别式探针</b>
（跑一个小程序、收 op 级 time 记录，P-oracle 本就允许）<b>解不了这里的死结</b>。
两个理由都是类型级的：<b>①</b> 探针是<b>全称命题的实例化</b>，只能证伪
<code>replication(framework_ver#rule)</code> 这类规则；分片表的根是
<code>abstraction(placement_declaration@spec_id)</code>，绑定到<b>那一份 spec</b>，
换个程序证明不了它。<b>②</b> 半自动并行的策略传播是一个<b>求解器</b>，其输出<b>非局部</b>
⇒ 探针 trace 在原理上不可外推。<b>③</b> 而且<b>以显存为观测量的探针在类型上不存在</b>
——那正是 <code>G-X4</code> 禁的东西。<b>探针是成本与灵敏度的改进，不是能力的扩张。</b></p></div>
```

影响：  L-057 **改首页字段**（新增一行）；L-030 只加正文段落，不改门表、不改门数。E12 与 E9 呼应（同一条"调宽映射表"的收益路径，一处给数、一处封坍缩权）。

---

### E13 · L-074(a) · 首页缺一行"两个头号输出均非界"

现状核实： §10.5 行 1703 有 `oracle 声明` 行、行 1706 有 `S_∞ ∩ live(t*)` 行并附"给不出装得下"的话 ✓，但**没有**"峰值区间的宽度不覆盖 `reconstructed` 类复刻的偏差"这条。§10.2 行 1478 已就 `.lo` 侧写了"它是一个条件下界，不是一个定理"，§13.3 行 2148 又写"已分配峰值有真正的区间（是界）"——两句都对，但读者只看首页时会取后一句。

判定：    采纳台账修法 (a)。理由：零成本，且它把 §10.2 已有的那句限定从章节深处提到首页；它与 §14.2 的 `size_unverifiable` 是同一件事的两种呈现（一个是数、一个是资格声明）。

锚点：
```
<tr><td class="nw"><code>oracle 声明</code></td><td>「显存链外部裁判：无。时间链外部裁判：op 级序列与时长。」</td><td>—</td></tr>
```
替换为：
```
<tr><td class="nw"><code>oracle 声明</code></td><td>「显存链外部裁判：无。时间链外部裁判：op 级序列与时长。」</td><td>—</td></tr>
<tr><td class="nw"><code>产物形态声明</code></td><td>「<b>本报告的两个头号输出都不是无条件的界</b>：<code>peak_interval</code> 是<b>给定复刻规则正确</b>条件下的区间——它的宽度<b>不覆盖</b> <code>replication</code> 规则本身判错的那部分偏差（<code>S_∞</code> 与 <code>size_unverifiable</code> 各管其一）；<code>step time</code> 只有 <code>Sample</code> 形态，是<b>扫描观测极值</b>，不是界。」<b>schema 约束，不可抑制</b></td><td>—</td></tr>
```

影响：  改首页字段（新增一行）。与 E5（`peak_allocated.lo` 的条件性）、§14.2 边界 E 耦合；**须在 E5 之后落盘**（否则"条件下界"这句话在正文里还没有精确对象）。

---

### E14 · L-053(残) · 档表里没有 B′ 行，四分穷尽的论证与表对不上

现状核实： §13.3 行 2121-2131 的 bad 框已按 L-053 论证"划分谓词改 `interval` 的来源 + 补一档 B′"✓，末句写"五个来源 + 兜底 ⇒ 每个量恰落一档"。**但紧随其后的档表（行 2133-2141）只有 A/B/C/D 四行，且 A/B 两行的内容列仍用 `certainty` 描述**（"`assumed` 的数值区间型根" / "`modeled` 的残差根"）⇒ 论证与表互相矛盾，穷尽性在表上不成立，而 §13.3 的 `2k → 34` 成本论证依赖穷尽。

判定：    自拟（补齐 L-053 的落地残缺）。理由：这是已落地条目的残留，不是新设计；台账修法本身已被裁定采纳，只是没写进表。

锚点：
```
<tr><td class="nw">A</td><td><code>assumed</code> 的数值区间型根（MoE 负载、capacity、稀疏选中数）</td><td class="nw">10–20</td>
  <td class="nw"><b>否</b>，联合取端点</td><td>已分配显存对字节<b>单调</b>（执行序由调度确定性产出，<b>不依赖任何字节数</b>）</td></tr>
<tr><td class="nw">B</td><td><code>modeled</code> 的残差根（~90 算子族 + workspace 域片）</td><td class="nw">~100</td>
  <td class="nw"><b>否</b>，联合取端点</td><td>同上；残差是相对乘性的，字节侧仍单调</td></tr>
```
替换为：
```
<tr><td class="nw">A</td><td><b>区间来自 <code>assumption</code> 根的端点</b>（MoE 负载、capacity、稀疏选中数）</td><td class="nw">10–20</td>
  <td class="nw"><b>否</b>，联合取端点</td><td>已分配显存对字节<b>单调</b>（执行序由调度确定性产出，<b>不依赖任何字节数</b>）</td></tr>
<tr><td class="nw">B</td><td><b>区间来自 <code>residual_rel</code></b>（来源③，~90 算子族 + workspace 域片）</td><td class="nw">~100</td>
  <td class="nw"><b>否</b>，联合取端点</td><td>同上；残差是相对乘性的，字节侧仍单调</td></tr>
<tr><td class="nw"><b>B′</b></td><td><b>区间来自 <code>declaration_unverified_bound</code></b>（来源④）：<code>G-B4</code> 判出的完美均衡量、§8.3 手写分片表的 degree、Tier 2 的 saved 字节、alias 等价类。<b>两端均由定义式算出</b>，与 A/B 同性质</td><td class="nw">视情况</td>
  <td class="nw"><b>否</b>，联合取端点</td><td>同上。<b>并入既有的 2 次 liveness，增量为零</b>；缺这一档则这些量<b>不落任何一档、没有扫描规程</b>，而 <code>2k → 34</code> 的成本论证依赖四分穷尽</td></tr>
```

影响：  不改门表、不改门数。与 E3（`G-B4`）、E6（`span`）同节；穷尽性论证从此与表一致。

---

### E15 · L-036(残) · §7.3 说"四条都是门"，§12.2/§12.3 说 `G-N4` 是后置条件

现状核实： §12.2 行 1991 已把 `G-N4` 标为**后置条件** ✓，§12.3 行 2008 已把它列进"降级为后置条件（5）"✓。**但 §7.3 行 1221 的引导句仍写**「四条机器可检的断言（<b>都是门，不是后置条件</b>——各自都有一个规格级缺陷能让它变红，见第 12 章）」，并明确指向第 12 章——而第 12 章正好否掉它。

判定：    自拟（L-036 落地残留）。理由：`G-M1` 的整套口径就建在"门 vs 后置条件"的分类上，同一份文档两处相反的分类，是它自己要抓的那种缺陷。

锚点：
```
（<b>都是门，不是后置条件</b>——各自都有一个规格级缺陷能让它变红，见 <a href="#c12">第 12 章</a>）：</p>
```
替换为：
```
（<b>其中三条是门</b>：<code>G-N5</code>/<code>G-N6</code>/<code>G-N7</code> 各有一个规格级缺陷能让它变红；
<b><code>G-N4</code> 按 <code>G-M1</code> 降为后置条件</b>——它对自己宣称要抓的东西恒真，见 <a href="#c12">第 12 章</a>）：</p>
```

影响：  不改门数（§12.2/§12.3 已按后置条件计）。纯一致性修复。

---

### E16 · L-067 · §11.1 阻断表的一行恒命中，区分力为零

现状核实： §11.1 行 1748 行标仍是「**已注册，含 `modeled` 量**」。而每个模型必然含 `modeled` 量（任何由阻断产生的注册项 + 全部 `MechanismOp`，§2.5 P2 已把 `MechanismOp.self_certainty` 钉死为 `{modeled}`）⇒ 该行恒命中，"越出域片 ⇒ 阻断"被读成普适条件。

判定：    采纳台账修法（R4-C C-11）。理由：与 §10.1（L-018 已落地）新写的"两种失效"完全对齐——那里已把"从未建立过标定域片"与"有域片但越域"分开，本行的条件必须同步改成后者。改后 `G-X3`（域片 := 凸包）使"越域"成为机器可判的谓词，行有了真实的触发条件。

锚点：
```
<tr><td class="nw"><b>已注册，含 <code>modeled</code> 量</b></td><td>允许执行；越出域片 ⇒ 阻断</td>
  <td>域片外没有标定点，残差无来源</td></tr>
```
替换为：
```
<tr><td class="nw"><b>某 <code>modeled</code> 量的参数<b>越出其标定域片</b></b></td><td>阻断</td>
  <td>域片外没有标定点，残差无来源。<b>行标不能写成"含 <code>modeled</code> 量"</b>——每个模型必然含
      （由阻断产生的注册项 + 全部 <code>MechanismOp</code>，其 <code>self_certainty</code> 值域就是
      <code>{modeled}</code>）⇒ 该行恒命中、区分力为零。<code>G-X3</code> 把域片改成凸包之后，
      "越域"是机器可判的谓词。注意与 §10.1 的另一种失效区分：<b>从未建立过域片</b>的算子族
      走 roofline 初始定价（合法，落 P5 兜底单侧），<b>不阻断</b></td></tr>
```

影响：  不改门表、不改门数。与 L-018、L-028（均已落地）耦合。

---

## 优先级 T2 —— 判据 5 反向失败 / 力度低于自称

---

### E17 · L-045 · "以指纹表登记"不可执行；最便宜的旁路只有纪律没有机制

现状核实： §0.1 行 66-70 已写死禁令文字 ✓，末句仍是「这一条必须在实现里以指纹表显式登记（`G-X4`）」。但**一个人肉抄进标定 YAML 的数字没有指纹**：指纹表能查"同一指纹出现两处"，查不到"这个数来自被禁来源" ⇒ 按判据 2（可达阻断性）不过，它是纪律不是机制。

判定：    采纳台账修法的**第一半**（R3-2 C7：在缺陷台账的 schema 上把 OOM 记录的字段裁剪为 `{config_digest, bool}`），**不采纳第二半**（"`CalibrationSet` 每点必填产出脚本 id"）。理由：脚本 id 是可以随手写的字符串，判据 1/2 都不过，它只是把纪律换了个位置；而 L-028 已落地的**凸包域片 + 留一交叉验证**已经把"抄一个数进标定"这条路堵死了——单点域片的凸包退化为一个点 ⇒ 目标配置必然越域 ⇒ 阻断。所以剩下的唯一缺口是**存储侧**：只要台账里能存下那些数，它们就会被读出来用。类型层消除是 §11.5 自己的方法论。

锚点：
```
这一条必须在实现里以指纹表显式登记（<code>G-X4</code>）。</p></div>
```
替换为：
```
这一条<b>不能只靠指纹表登记</b>——一个人肉抄进标定 YAML 的数字<b>没有指纹</b>；
指纹表能查"同一指纹出现两处"，查不到"这个数来自被禁来源"，那是纪律不是机制（判据 2 不过）。
<b>正确的走法是在类型层让它无处可存</b>：缺陷台账中 OOM 记录的 schema 字段
<b>裁剪为 <code>{config_digest, oom: Bool}</code></b>，不设任何数值字段
——<code>tried to allocate X</code> 这些数在 schema 上<b>没有落点</b>。
另一半由 <code>G-X3</code> 兜住：抄进 <code>CalibrationSet</code> 的孤点，其凸包域片退化为一个点
⇒ 目标配置必然越域 ⇒ 阻断。<b>能在类型层消除的，就不留给纪律。</b></p></div>
```

影响：  改缺陷台账 schema（新增约束）；不改门表、不改门数。与 L-028、L-046（均已落地）耦合。

---

### E18 · L-037 · §8.3 自相矛盾：`axis` 既"不改字节"又"改变通信种类"

现状核实： §8.3 行 1327-1328 的表写「切错轴**通常改变通信种类**（AllGather↔ReduceScatter）；若两轴等长且通信种类相同则抓不到」；行 1335 的代码块写「`axis` 侧 = 不进字节区间（切哪一轴不改字节），风险由 op 序列门承担」。两者不能同时为真——改变通信种类就改变了**该处插入的通信节点**的字节与其临时 buffer、以及 `Free` 时刻。§14.3 行 2269-2270 已披露"无对照 trace 的部署下无验证手段"✓，但字节侧的记账未改。§13.3 档 D 又把 `axis` 判进"不进区间扫描" ⇒ 声明 `axis` 在无 trace 部署下**零代价**。

判定：    采纳台账并列修法中的 **R4-C A-7**（改写措辞并点名承担者），不单独执行 R3-1 A8-b（给受影响的通信 buffer 挂 `abstraction(placement_declaration)` 根）。理由：A8-b 描述的根继承**已由 P4 蕴含**——通信节点是否存在由 placement 失配决定，而失配由 `axis` 决定，`axis` 带 `abstraction(placement_declaration@spec_id)` 根（行 1333 已写），按 P4「存在位下传」这些字节本就继承该根。真正缺的是**把这件事说对**：正文当前那句"不改字节"会让实现者照字面把这些字节的根切掉。写清即可，不需要新构件。

锚点：
```
           axis   侧 = 不进字节区间（切哪一轴不改字节），风险由 op 序列门承担
```
替换为：
```
           axis   侧 = **不改变被切张量本身的字节**，但**改变该处插入的通信节点的种类**
                       （AllGather↔ReduceScatter）⇒ 改变该节点的通信字节、临时 buffer
                       与 Free 时刻 ⇒ **改变峰值**。写成"切哪一轴不改字节"是错的。
                       这部分字节的存在位由 P4 继承 abstraction(placement_declaration@spec_id)
                       ⇒ 它们**进 θ_abs 的分子**，声明 axis 不是零代价的。
                       至于"切错了轴"这件事本身：由 G-T1 承担（通信种类不同 ⇒ op 身份不同）；
                       **无对照 trace 的部署下无任何承担者**（§14.3），且即使有 trace，
                       两轴等长且通信种类相同时仍看不见
```

影响：  不改门表、不改门数。改 §10.4 `θ_abs` 分子的实际覆盖面（变大，但仍不含 `replication` ⇒ 不触 §6.1c 裁定 #7）。§13.3 档 D 保持不变（`axis` 仍是离散清单，只是不再"零代价"）。

---

### E19 · L-073 · `residual_rel` 与 `assumed` 量的五元组未判定，两处接链断裂

现状核实： §2.5 行 375 `③ 标定残差 residual_rel （grade = calibrated）` ——`residual_rel` 自身**无 provenance**，而它是来源③ 的产物 ⇒ P1 在③ 上断链。§2.3 行 264 只写 `not_applicable 专给 certainty = exact 的量`，**`assumed` 量的 `evidence.grade` 未定** ⇒ §2.7/§10.5 那两行"按 `evidence.grade` 三行、划分、守恒"对 `assumed` 量无值可归 ⇒ 守恒不成立。（L-073 的其余子项已被间接解决：叶子 `certainty` 由 L-015 的叶子规则给出、`StorageRelation` 由 L-008 的 `replication(alias_rules)` 给出、`RepeatRegion.count` 在 §4.3 已是 `Q[Int]`。）

判定：    采纳台账修法，但**只取两条被标为"接链必要项"的**。理由：其余子项已有落点，逐条重写会把一节改成清单；这两条各自使一条已声明的守恒/传播律不成立，属必修。

锚点（1/2）：
```
                 ③ 标定残差 residual_rel                                （grade = calibrated）
```
替换为：
```
                 ③ 标定残差 residual_rel                                （grade = calibrated）
                    residual_rel 自身是一个 Q：roots ∋ calib(该量的 CalibKey)、certainty = modeled、
                    evidence.grade = calibrated。**不给它 provenance，P1 在来源③ 上就断链**
                    （由标定点算出的残差会成为一个无根的数）
```

锚点（2/2）：
```
            not_applicable 专给 certainty = exact 的量（它们的 evidence 是 ⊥）。
```
替换为：
```
            not_applicable 专给 certainty = exact 的量（它们的 evidence 是 ⊥）。
            **certainty = assumed 的量其 grade 一律 reconstructed**（其区间来自来源④ 的定义式，
            不来自任何实测或厂商文档）——不钉死这一格，"按 evidence.grade 三行、划分、守恒"
            对 assumed 量无值可归，那条守恒性当场不成立。
```

影响：  不改门表、不改门数。恢复 §2.7 与 §10.5 两处"划分，守恒"的成立条件。

---

### E20 · L-066 · `timing_dependent` 的"三个消费者"里第一个是它自己的产生规则

现状核实： §9.2 行 1379 已写"它就是 §2.5 的 P4（存在位下传）在 `Free` 事件上的实例，**不是第二个机制**"；行 1381 紧接着把"P4 的根继承"列为**消费者①**。同一段里同一件事既是产生规则又是消费者。

判定：    采纳台账修法（R3-3 §2.6），取"改为两个消费者"的写法。理由：另一种写法（把第一条改述为"它的产生规则同时给出根继承"）保留三个数，但那个数本身没有承重作用；两个消费者更短且不再自相矛盾。

锚点：
```
<p><b>它有三个消费者</b>（缺一它就退回标签）：<b>①</b> P4 的根继承——受影响 storage 在峰值时刻的存活成员资格
继承标定根；<b>②</b> 敏感性扫描的维度集合（维度不是"每个 kernel 的 duration"，
而是每条 <code>timing_dependent</code> 边所依赖的通信算子的 <code>CalibKey</code> 族）；
<b>③</b> 归因 liveness 的次数从 2 变 4（§2.7 四角）。</p></div>
```
替换为：
```
<p><b>它有两个消费者</b>（缺一它就退回标签）：<b>①</b> 敏感性扫描的维度集合（维度不是"每个 kernel 的
duration"，而是每条 <code>timing_dependent</code> 边所依赖的通信算子的 <code>CalibKey</code> 族，
<code>measured_quantity = duration</code>）；<b>②</b> <code>peak_allocated.hi</code> 的构造
——每条 <code>timing_dependent</code> 边把受影响 storage 的存活区间<b>取并</b>（§10.2）。</p>
<p><b>不把"P4 的根继承"算成消费者</b>：它是本投影<b>自己的产生规则</b>（上一段刚写完"它就是 P4 在
<code>Free</code> 事件上的实例"）。把产生规则算进消费者清单，会让"缺一它就退回标签"这句话
少一个真实的支点。</p></div>
```

影响：  不改门表、不改门数。**与 E7 耦合**（消费者①的措辞与档 C 的 `CalibKey` 族口径必须一致）；②已按 §10.2 的结构性上界改写，不再是"四角"。

---

## 优先级 T3 —— 措辞、命名与记账（不改机制，改可读性与口径）

---

### E21 · L-060 · `policy` 已是根上的谓词，§1.3/§1.4 仍写成根

现状核实： §2.3 行 258-261 已改为谓词 ✓；§1.3 行 137 与 §1.4 行 179 仍写 `∀f ∈ G0: policy(·) ∉ f.roots`。注意 §12.1 行 1860 已用新写法 `∀f ∈ G0 : ¬policy(f)`，三处口径不一。

判定：    采纳台账修法（机械替换）。

锚点（1/2）：
```
  <td><b><code>∀f ∈ G0: policy(·) ∉ f.roots</code></b>；<code>digest(G0)</code> 在策略扰动下不变</td>
```
替换为：
```
  <td><b><code>∀f ∈ G0: ¬policy(f)</code></b>（<code>policy</code> 是根上的谓词，不是第九种根，见 §2.3）；<code>digest(G0)</code> 在策略扰动下不变</td>
```

锚点（2/2）：
```
<li><b>policy-free 核</b>：<code>∀f ∈ G0: policy(·) ∉ f.roots</code>；策略只能由 pass 重新引入。</li>
```
替换为：
```
<li><b>policy-free 核</b>：<code>∀f ∈ G0: ¬policy(f)</code>；策略只能由 pass 重新引入。</li>
```

影响：  无。纯措辞。

---

### E22 · L-061 · guard 四分类占用 `G-*` 门命名空间

现状核实： §4.2 行 820-823 的四个类名 `G-irrelevant` / `G-reconstructible` / `G-absorbable` / `G-semantic`；行 835 正文再次引用两个。而 §12.2 行 1870 的立论是"前缀本身携带信息：一眼看出这道门拿什么当裁判"，四者是 guard **分类**、无 oracle 归属；§12.1 又刚新增 `G-A*` 前缀，命名空间更紧。

判定：    采纳台账修法（改前缀为 `PG-*`）。`BLK-*` 不动（本就是另一个命名空间）。

锚点（1/2，四行连续）：
```
<tr><td class="nw"><b>G-irrelevant</b></td><td>两支残差同构（如日志/断言分支）</td><td><code>digest(res_then) == digest(res_else)</code></td><td>降级到下三类继续判</td></tr>
<tr><td class="nw"><b>G-reconstructible</b></td><td><b>非-π₀ 支的残差正是某个 pass 会插入的东西</b></td><td><b>PGRO</b>（见下）</td><td><code>BLK-UNRECONSTRUCTIBLE</code></td></tr>
```
替换为：
```
<tr><td class="nw"><b>PG-irrelevant</b></td><td>两支残差同构（如日志/断言分支）</td><td><code>digest(res_then) == digest(res_else)</code></td><td>降级到下三类继续判</td></tr>
<tr><td class="nw"><b>PG-reconstructible</b></td><td><b>非-π₀ 支的残差正是某个 pass 会插入的东西</b></td><td><b>PGRO</b>（见下）</td><td><code>BLK-UNRECONSTRUCTIBLE</code></td></tr>
```
（`G-absorbable` / `G-semantic` 两行同法逐字改为 `PG-absorbable` / `PG-semantic`；此二锚点见下）

锚点（2/2）：
```
<p><b><code>Π_extent</code> 的 guard 绝大多数落在 G-reconstructible，而不是 G-irrelevant。</b>
```
替换为：
```
<p><b><code>Π_extent</code> 的 guard 绝大多数落在 <code>PG-reconstructible</code>，而不是 <code>PG-irrelevant</code>。</b>
```

补充：建议在四分类表前加一句「<b>前缀是 <code>PG-*</code> 不是 <code>G-*</code></b>：<code>G-*</code> 按 §12.2 的规范表示"门"，前缀携带 oracle 信息；guard 四分类是<b>分类</b>，没有 oracle、没有 pass/fail 出口。」——插入锚点为四分类表的 `<tr><th class="nw">guard 类</th>` 行之前的 `<div class="tw"><table>`（该 `div` 在文中不唯一，落盘时请以行 818 定位）。

影响：  不改门表、不改门数（这四个本就不在门表里）。**唯一的落盘风险**：必须确认没有别处引用旧名（已核实全文仅 5 处，均列于上）。

---

### E23 · L-062 · 禁用词自违五处

现状核实： §0.1 行 48 的禁令仍在。五处仍在：行 419（P8「验证：对闭原语集…」）、行 908（§4.4 步骤6「校验 G0 不变量」，含 alias 链无环、每输出恰一条存储规则 —— 正是显存链）、行 1086（§6.7 步骤4「值域裁剪校验」）、行 1761（§11.2 步骤3「校验」）、行 1927（§12.2 `G-E3`「随机取样验证」）。

判定：    采纳台账并列修法中的**逐处改写**——这一支已由 ADVERSARIAL-RECORD §6.1 裁定 #10 选定，理由是"在 §0.1 给禁令加'按语境判断'的按语，与 §11.5「按名字禁是禁不住的，按形态才行」的立论自相矛盾"。本文照此执行，不重开裁定。
（另：行 623/762/771/903 的"配置校验断言"是**框架自己源码里的 assert** 的专名，行 935/1254/2262-2278 等是否定式"无验证手段"，均在禁令射程之外，不动。行 2167「注册、校验、解释」与行 2179「可静态校验」建议同批改为"检查"，属可选。）

锚点与替换（五处，逐条）：

1. `               验证：对闭原语集（约 15 条）做 property test，随机取样断言 Prov(f(x)) ⊒ ⊔ᵢ Prov(xᵢ)`
   → `               自检：对闭原语集（约 15 条）做 property test，随机取样断言 Prov(f(x)) ⊒ ⊔ᵢ Prov(xᵢ)`

2. `<li>校验 G0 不变量：<b>无 policy root</b>、无 opaque、每输出恰一条存储规则、alias 链无环无悬空、side-effect 算子未被跳过。</li>`
   → `<li>逐条断言 G0 不变量：<b>无 policy root</b>、无 opaque、每输出恰一条存储规则、alias 链无环无悬空、side-effect 算子未被跳过。</li>`

3. `<li><b>值域裁剪校验</b>：<code>RegisteredOp.self_certainty</code> 取 <code>exact</code> <b>仅当</b>`
   → `<li><b>值域裁剪检查</b>：<code>RegisteredOp.self_certainty</code> 取 <code>exact</code> <b>仅当</b>`

4. `<li>用户补充完整 → 校验（含值域裁剪、<code>CalibKey</code> 解析）→ 解释（打印九个面的完整语义，供人工复核）。</li>`
   → `<li>用户补充完整 → 检查（含值域裁剪、<code>CalibKey</code> 解析）→ 解释（打印九个面的完整语义，供人工复核）。</li>`

5. `<td>单调性 property test（每个算术原语的声明单调性经随机取样验证）</td>`
   → `<td>单调性 property test（每个算术原语的声明单调性经随机取样断言）</td>`

影响：  无。纯措辞，但它是 §0.1 自己的禁令，留着就是文档级的自违。

---

### E24 · L-063 · `Evidence.sweep` 无消费者

现状核实： 行 263 `Evidence := { grade ∈ {...}, ref, sweep }`、行 975 `evidence{grade, ref, sweep}`。全文再无第三处提到该字段（`sweep_dims` / `excluded_from_sweep` / 敏感性扫描均为另一件事，已逐条核对）。

判定：    采纳台账并列修法中的**删除**一支。理由不止"无消费者"：`G-X3`（已落地）刚把**域片**从"声明的"改成"由标定点集导出的"，而一个可声明的、名叫 `sweep` 的证据字段正是那条路的复活形态——留着它，"我们扫过了/我的证据覆盖这个域"就有了一个 schema 落点。删掉同时使 §11.4 `G-X1` 的闭集断言更紧。

锚点（1/2）：
```
Evidence  := { grade ∈ {calibrated &lt; vendor_documented &lt; reconstructed, **not_applicable**}, ref, sweep }
```
替换为：
```
Evidence  := { grade ∈ {calibrated &lt; vendor_documented &lt; reconstructed, **not_applicable**}, ref }
            **无 sweep 字段**：它没有消费者，而一个可声明的"我的证据覆盖这个域"正是
            G-X3 刚把域片改成导出量时封掉的那条路的复活形态
```

锚点（2/2）：
```
<tr><td class="nw"><b>epistemics</b></td><td><b><code>self_certainty</code> + <code>evidence{grade, ref, sweep}</code></b>（§2.5 P2/P3）</td><td>provenance 代数</td><td>自证面</td></tr>
```
替换为：
```
<tr><td class="nw"><b>epistemics</b></td><td><b><code>self_certainty</code> + <code>evidence{grade, ref}</code></b>（§2.5 P2/P3）</td><td>provenance 代数</td><td>自证面</td></tr>
```

影响：  改 schema（删字段）；`G-X1` 的闭集断言随之更紧。§6.3 的示例（行 997）只写了 `{grade: reconstructed}`，无需改。

---

### E25 · L-064 · §8.2 表把 `GatherParam` 与 `FreeParam` 挤在一行、root 写"同上"

现状核实： 行 1292-1293 该行 root 列写"同上"（指向 `replication(bucketing_rule)`）——**分桶规则不管 gather/free 的插入位置**，指代错；且二者可验证性完全相反（前者留真实设备算子、可被 `G-T1` 约束；后者是主机侧动作、在 `O` 中不存在），却共用一行 provenance 与同一个 `evidence.grade`。同节后文行 1309-1316 已把这个差别论证清楚 ✓，表未拆。

判定：    采纳台账修法（R4-C C-3/C-4）。

锚点：
```
<tr><td><code>GatherParam</code> / <code>FreeParam</code> 的<b>插入位置</b></td>
  <td class="nw">同上</td><td class="nw"><code>modeled</code></td><td>它不能由 placement 失配推出，是实现策略</td></tr>
```
替换为：
```
<tr><td><code>GatherParam</code> 的<b>插入位置</b></td>
  <td class="nw"><code>replication(gather_rule)</code></td><td class="nw"><code>modeled</code></td>
  <td>它不能由 placement 失配推出，是实现策略。<b>留真实设备算子</b> ⇒ 可被 <code>G-T1</code> 约束</td></tr>
<tr><td><code>FreeParam</code> 的<b>插入位置</b></td>
  <td class="nw"><code>replication(lifetime_rule)</code></td><td class="nw"><code>modeled</code></td>
  <td>同上，但<b>可验证性正好相反</b>：归还显存是主机侧动作，在 <code>O</code> 中<b>根本不存在</b>（见下）
      ⇒ 与 <code>GatherParam</code> 不能共用一行 provenance、也不能共用一个 <code>evidence.grade</code>。
      <b>root 不是"同上"</b>——分桶规则不管插入位置</td></tr>
```

影响：  不改门表、不改门数。与 §2.5 P4 表（`Pass_sched → replication(lifetime_rule)`，已落地）口径一致。

---

### E26 · L-065 · Tier 表用旧名 `derived`，且 `captures 集合` 与 `captures 导致的字节` 同名

现状核实： §7.2 行 1162-1166 的 Tier 块首行仍是 `Tier 1 · derived      captures 的 roots 含 source(framework)`。两个问题：(a) `derived` 现已是 `certainty = exact` 的命名投影，用作 Tier 标签会被读成类型；(b) 更实质的是**同名两个量**——`captures` **集合**（源码文本解析结果，`source` 根、可 `exact`）与 `captures` 导致的**字节**（liveness 成员资格，由 `Pass_sched` 产出、按 P4 继承 `replication(lifetime_rule)`、必为 `modeled`）。混同会让这批字节被记成 `exact`，直接错算 `c1` 与 `evidence.grade` 三行分解。

判定：    采纳台账修法（R4-C C-12 + R3-3 §3.14），两处一起改。理由：(b) 是记账错误而非措辞——它改变首页两个守恒分解的值。

锚点：
```
<pre><code>Tier 1 · derived      captures 的 roots 含 source(framework)
                      **G-D4 生效**：template_free_vars ∩ forward_values == capture_site_args
```
替换为：
```
<pre><code>Tier 1                **captures 集合**：roots ∋ source(framework)，certainty = exact
                      **captures 导致的字节**：由 Pass_sched 的 liveness 产出，按 P4 继承
                          replication(lifetime_rule) ⇒ **certainty = modeled**
                      —— 两个量必须分列。混同会把这批字节记成 exact，
                         直接错算首页的 c1 与 evidence.grade 三行分解
                      **G-D4 生效**：template_free_vars ∩ forward_values == capture_site_args
```

影响：  不改门表、不改门数。改首页 `按 certainty 分解` 与 `按 evidence.grade 分解` 两行的取值（更诚实）。

---

### E27 · L-068 · `abstraction_ledger` 三列对新形态无定义

现状核实： §10.5 行 1711-1712 `abstraction_ledger` 列为 `judgment_id, kind, readability, coverage_bytes, residual_width, 首次引入于哪个模型`，只禁了 `residual_width` 打印裸 `0`。未定义：(a) 无机器来源的条目 `residual_width` 填什么；(b) `kind` 列对 `declared_semantics` 条目（§2.3 已给它一个**另一个**闭集：`{fused_kernel, captures, allocator, collective}`）无定义；(c) `readability` 列同理。

判定：    采纳台账修法（R3-1 N5）。理由：零成本，且它正是 L-025（已落地的 `U` 定义）之后 `size_unverifiable` 的**逐条明细**——没有它，首页那个占比无法下钻。

锚点：
```
<tr><td class="nw"><code>abstraction_ledger</code></td><td>每条人工判断一行：<code>judgment_id, kind, readability, coverage_bytes, residual_width, 首次引入于哪个模型</code>；
  <b>残差列禁止打印裸 0</b>，必须写 <code>0（P_test 覆盖 N 点，非证明）</code></td><td>—</td></tr>
```
替换为：
```
<tr><td class="nw"><code>abstraction_ledger</code></td><td>每条人工判断一行：<code>judgment_id, root_kind, kind, readability, coverage_bytes, own_coverage_bytes, residual_width, 首次引入于哪个模型</code>。
  <b><code>root_kind</code> ∈ {<code>abstraction</code>, <code>declared_semantics</code>}</b>，<code>kind</code> 列<b>随之取两族闭集之一</b>（§2.3 各有各的闭集）；<code>readability</code> 列对 <code>declared_semantics</code> 条目恒为 <code>NONE</code>（按 §2.4 分派表，这是它入表的条件）。
  <b>残差列禁止打印裸 0</b>：有构造性残差者写 <code>0（P_test 覆盖 N 点，非证明）</code>；<b>无机器来源者写 <code>不可计算（无源码，evidence.grade = X）</code></b>，禁止留空、禁止填 0。
  本表是 §14.2 <code>size_unverifiable</code> 的<b>逐条明细</b>——没有它，首页那个占比无法下钻</td><td>—</td></tr>
```

影响：  改首页字段（列变化）。与 L-004（已落地的 `own_coverage`）、L-025（已落地的 `U`）耦合——`own_coverage_bytes` 一列即 §10.4 要求"并列打印第二列"的落点。

---

### E28 · L-056 · 首页统计字段自身无 provenance

现状核实： §10.5 全表无 provenance / `presentation` 标注；§2.7 行 573 只给"按 root 种类"这一个聚合标了 `presentation: not_a_partition`。而 §10.4 拿 `judged` / `base` 去与 Θ 比较，§11.3 拿 `c1..c5` 进 `L-mono` 的偏序。

判定：    采纳台账修法，但落成**一条对全表生效的规定**而不是逐字段加列。理由：逐字段加列会把首页表撑成四列；一条规定同时防住真正的风险——有人把这些统计量当作 `Q` 喂回链里（它们没有 `interval`、没有 `direction`，一旦进区间算术就是无声的伪精确）。

锚点：
```
<div class="cal why"><span class="lbl">归因口径的三条论证</span>
```
替换为：
```
<div class="cal inv"><span class="lbl">首页统计字段自己也要有身份</span>
<p>上表的每一个统计字段（<code>judged</code> / <code>base</code> / <code>unverifiable_coverage</code> /
<code>clean</code> 五分量 / <code>new_lexicon(k)</code> / <code>bwd_ambiguous_ops</code> /
<code>op_identity_map_load</code>）在 schema 中<b>必须标 <code>presentation</code></b>，
并<b>打印它依赖的根集合</b>（= 参与其求和的那些量的 <code>roots</code> 之并）。</p>
<p><b>它挡的是什么</b>：这些是<b>呈现量</b>，没有 <code>interval</code>、没有 <code>direction</code>。
一旦有人把它们当作 <code>Q</code> 喂回链里（"用 <code>judged</code> 反推一个修正系数"），
就得到一个无声的伪精确值——而 <code>G-X1</code> 只查字段名与枚举值，查不到这个形态。
标 <code>presentation</code> 之后，它们进区间算术即为类型错误。</p>
<p><b>连带</b>：§10.4 拿 <code>judged / base</code> 与 Θ 比较是合法的（阈值比较不是区间算术），
但报表必须同时打印这两个数各自的依赖根集合，否则"Θ 触发了"这句话无法归因。</p></div>

<div class="cal why"><span class="lbl">归因口径的三条论证</span>
```

影响：  改 schema（`presentation` 的射程）与首页呈现。与 E12（新增 `op_identity_map_load`）耦合——须在 E12 之后落盘，否则清单里少一项。

---

### E29 · L-055(残) · "`S_P` 不过 ψ"未进门表

现状核实： §2.6 行 523-529 已把理由写全 ✓（这是 L-055 的主体）。台账修法末句要求"这一句必须进门表"，而 §12.2 行 1940-1941 的 `G-B1a` 行未提 ψ。

判定：    自拟（补齐 L-055 落地残缺）。理由：门表是实现者唯一会照抄的清单；`G-C3` 的右边套 ψ 就在同一张表上，隔了几行，照抄时套上 ψ 的概率很高，而后果是 `hull(0, X)` 对每一个做通信的被抽象模块恒成立。

锚点：
```
<tr><td class="nw"><code>G-B1a</code></td><td>抽象残差：<code>readability=FULL</code> 时 shape/arity 在 <code>R_P</code> 与 <code>S_P</code> 上一致，
  其余各面取凸包作区间</td><td>人声明的语义与源码直接矛盾</td><td class="nw">门（<b>抽样，非证明</b>）</td></tr>
```
替换为：
```
<tr><td class="nw"><code>G-B1a</code></td><td>抽象残差：<code>readability=FULL</code> 时 shape/arity 在 <code>R_P</code> 与 <code>S_P</code> 上一致，
  其余各面取凸包作区间。<b><code>S_P</code> 不过 <code>ψ</code></b>（与同表的 <code>G-C3</code> 不同）——套上 ψ 会删掉全部通信节点
  ⇒ <code>S_P.通信字节 = 0</code> ⇒ <code>hull(0, X)</code> 对每一个做通信的被抽象模块<b>恒成立</b>，
  把正确的抽象罚成宽区间。<b>不投影是它比 <code>G-C3</code> 强的地方，不是疏漏</b>（§2.6）</td><td>人声明的语义与源码直接矛盾</td><td class="nw">门（<b>抽样，非证明</b>）</td></tr>
```

影响：  不改门数。纯口径补齐。

---

# 3. 需要裁决者拍板的

## 3.1 E11 新增 `G-C3b` —— 是否接受门数从 40 变 41

**分歧点**：`G-C3b` 是真门还是 `G-E8` 的重复。
- **支持新增**：`G-E8` 断言 `used_fields ⊆ 授权集`，**少传**只会让 `used_fields` 更小 ⇒ 对"少传一个字段"恒过；而"少传"恰是 §3.5 删掉 `PolicyDerived` 侧表之后**唯一失去比对物**的那一类错误。按 §12.3 自己的判据（能否构造缺陷使 A 红 B 绿），构造是现成的：pass 只传 `vocab` 不传 `tp`，`used_fields = {vocab} ⊆ 授权集` ⇒ `G-E8` 绿；而全量 PE 轨迹在同一 anchor 处的环境含 `tp` ⇒ `G-C3b` 红。
- **反对新增**：它要求把"全量 `PE(Src,·,P)` 的求值轨迹"物化并按 anchor 索引，这是一份新的编译期制品（体积随 anchor 数线性），且 §13.1 的对账表未计这项成本。
- **两边代价**：不加 ⇒ §14.2 的能力边界要多一条"pass 现场调 PE 时少传字段无验证手段"；加 ⇒ 门数与首页三个切分全部重算，且 §13.1 R21 的遍数需要重新论证是否多一遍。

**本文的倾向**：加。因为 §3.5 删侧表时给出的理由是"现场调 PE 没有可被手写的中间制品"，而它换来的代价（丢掉比对物）当时没有被记账。但**这条必须由裁决者点头**，因为它是本轮唯一改门数的条目。

## 3.2 E1 的切分线：`transcribed` 的 `EnvProbe` 算不算"测量"

**分歧点**：`EnvProbe` 的执行结果使该条 `transcribed` 从 `modeled` 升为 `exact`（§2.5 P2 叶子规则）⇒ `c1` 上升。按 E1 的切分它属"测量类"、不受 `L-mono` 约束。
- **反对**：`EnvProbe` 是**我们写的探测脚本**，不是外部实测；写一个恒返回期望值的 probe 是零成本的，于是"加 probe ⇒ `c1` 上升"成为一条免罚的答案美化通道。
- **支持**：probe 进快照、其执行结果参与 digest（§3.1 已落地），改 probe 会改 digest 上首页；且它探的是 HBM 容量、设备型号这类**当场可核**的事实。
- **两边代价**：划进测量类 ⇒ 多一条理论上的美化通道；划进声明类 ⇒ 给 `EnvFacts` 写 probe 这个明确该被鼓励的动作要付 `clean` 代价。

**本文的倾向**：划进测量类，但要求 `EnvProbe` 的**执行环境指纹**与 `G-T0b` 同源（同一套框架/运行时/设备指纹）。不过这一句已经超出"编辑规格"的范围，请裁决者定。

## 3.3 E23 的射程：行 2167 / 2179 的两处"校验"改不改

裁定 #10 已选"逐处改写"，但 R4-C 只清点了五处。行 2167「开放用户通道：注册、**校验**、解释、骨架生成」与行 2179「可静态**校验**」也是我们自己的动作。改则彻底，不改则留两处半自违。**代价都极小，请一次定死射程**，免得下一轮再清点一遍。

---

# 4. 判定为「不改」的条目及理由

| 条目 | 理由 |
|---|---|
| **L-049(a)**（默认拒绝的假阳性） | 台账修法是把阻断点从"读值流入残差"移到"流入**语义面**（shape/dtype/整除判定/guard 条件）"。**但那份清单为了不丢覆盖必须包含广延量**——`_STATE['pp'] → num_layers_per_stage → RepeatRegion.count` 决定字节却不落在四项里。一旦补进广延量，"语义面"与"流入残差"就同外延，改法零收益；而它多出一个**可被少列一项**的闭集，正是 §11.5 判为逃逸的那种形态（少列一项 ⇒ 门更绿）。另一方面判据 3 已有实测支撑（§3.4：强可变全局 24/143，闭集可枚举）。**假阳性的代价是真的，但低于新增一个可被少列的清单。** 残余风险（schema 接不住"声明为无关"，一次假阳性会逼用户写一条假映射）建议记进 ADVERSARIAL-RECORD，不进正文。 |
| **L-069**（门数重复计数） | 台账的前提是"共用 red fixture ⇒ 不计入独立门数"。**正文的口径不是这个**：§12.3 行 2044-2050 明写"数门的正确判据是「能否构造一个缺陷使 A 红而 B 绿」，不是「它们读同一份数据吗」"。两对都能构造：`G-N6` 红而 `G-C1`③ 绿 —— pass 正确地新增了 Storage（可逆、`strip_synth` 能还原）但会计上把原 Storage 也算成已释放；`G-N7` 红而 §6.5 绿 —— op 正确声明 `has_internal_quant_buffer = false`，是 `Pass_precision_fwd` 漏建 `amax` 存储。**且 §6.5 的 `has_internal_quant_buffer` 是加载期必填项，根本不在门表里、不参与门数**。⇒ 不是缺陷。 |
| **L-075**（两条报告项落地） | 二者都是**报告项**：无阈值、无后果，判据 4 明确不过。它们唯一的价值是被人拿去做判断，而正文没有任何机制阻止读者用它们去坍缩区间——`Delta[T]` 更是被 R4-2 §4 判为**一等逃逸口**（B 由用户选，选一个跑通过的基线就能把 `undetermined` 读成 `false`）。引入一个"存在但被禁止用于 X"的**新类型**，与 §11.5「能在类型层消除的先消除」正面相反。`same_stream_gap_profile` 的机制前提已被判为假（排队深度不在 `O` 里）。**若将来要落地，前置条件是 E9 的坍缩权等价类表先存在**——那时"不得用于坍缩"才有机器执行体。 |
| **L-074(b)**（`G-T1` 通过 ⇒ 峰值坍缩到观测线性化） | 明确否决，理由已写进 E9 的替换文本：(1) `G-T1` 是**保序子序列**断言且带"明示不建模族"的计数豁免，通过它**不蕴含**真机线性化唯一；(2) 它会使"匹配上 trace"变成一个让答案变好看的动作，而让 `G-T1` 更容易匹配的办法正是调宽 op 身份映射表 —— §14.2 已披露那会让最强的门静默退化为恒真门 ⇒ **判据 5 反向失败**；(3) 无对照 trace 的部署下它给不出任何东西，收益不对称。 |
| **L-031 之 (2)** 的枚举式形态 | R4-1 要求"在洁净 fixture 上**枚举全部**使 verdict 变确定的填法路径 `p`，断言 `argmin_p cost(p)` 同时使 `c1` 或 `c5` 严格上升"。枚举是组合的，且 `cost(p)` 把"来源④条目数"算成脏化——而来源④ 现已是**一切对世界的断言**的标准形态（§2.5 P5，L-009 已落地），把它计入脏化会惩罚正确用法。**(2) 想要的性质由 E1 的切分直接给出**：使 `G-M2b` 变绿只有两条路——去测量（测量类，不受 `L-mono` 约束，代价是真的去测）或去声明（声明类，`L-mono` 强制至少一个分量严格下降）。不需要枚举。 |
| **L-030 的 `ProbeSuite` / `G-T5` 构件** | (1) 其承重论证（`G-T0b` 会判掉录制 trace ⇒ `G-T` 家族集体失 fixture）**不成立**：`G-M1` 的 fixture 是缺陷注入 fixture，其 `EnvFacts` 与 trace 同属该 fixture、自洽；叠加 E4 允许合成 op 序列后，`G-T` 家族的 red fixture 根本不需要真机 trace。(2) 探针**不是能力扩张**——"跑一个小程序收 op 级 time 记录"P-oracle 本就允许，它是一条工程实践，不是设计构件。(3) **以显存为观测量的探针在类型上不存在**（`G-X4`）⇒ 它对本方案最缺裁判的那一半（显存链）不产生任何新裁判。**但它的定性禁令必须进正文**（已落 E12），因为 ADVERSARIAL-RECORD §3.3 明确要求防止有人拿它复活 §8.3 的死结。 |

---

# 5. 落盘顺序与耦合

## 5.1 硬顺序（前者不落，后者的文本会写错）

```
E11 (新增 G-C3b)  ──→  E8 (§12.3 三个数)          # 门数必须一次算完再写
E5  (peak_allocated.lo 证据加权划分) ──→ E13 (首页"非界"行)   # "条件下界"要先有精确对象
E12 (op_identity_map_load) ──→ E28 (首页统计字段清单)         # 清单里要含新字段
E1  (L-mono 测量/声明二分) ──→ E4 (c5 定义与 G-B0 fixture)    # c5 的判据 5 论证建在 E1 上
E22 (PG-* 改名) 放在最后                                       # 与 E10 无冲突，但避免锚点漂移
```

## 5.2 会互相影响门数的

- **只有 E11 改门数**。落盘后 §12.3 的完整新数（E8 按此写）：
  `条目 49 · 后置条件 5 · 报告项 1 · 真门 43 · 独立性折扣 2 · 独立真门 41`；
  oracle 分布 `schema 6 / 图与源码文本 17 / 快照 4 / 真机 op time 5 / provenance 2 / 实现与规模 5 / 元判据 4 = 43`；
  三个切分 `oracle 在工具之外 5（其余 38）· 其中检模型事实 2 · 其中相互独立 1`；
  另一刀 `检模型事实 16 : 自律门 27 = 43`（`G-C3b` 归自律门·求值纪律，与 `G-E7a/b`、`G-E8`、`G-P3` 同类）。
- E9（坍缩权）、E12（探针定性）**刻意设计为不改门数**——前者改一条既有规则的边界，后者只加论证段落。这是选它们而不选 `G-T4` / `G-T5` 的理由之一。

## 5.3 同段/同表，落盘时会互相挤位置的

| 位置 | 涉及条目 |
|---|---|
| §2.3 类型集合 `pre` 块 | E19(2/2) · E24(1/2) |
| §2.5 P5 `pre` 块 | E19(1/2) |
| §3.2 三类策略字段表 | E10 |
| §3.3 `PolicyClassTable` `pre` 块 | E2 |
| §4.2 guard 四分类表 | E22 |
| §8.3 `pre` 块（相邻两行） | E18（`axis` 行）· E9（`坍缩条件` 行） |
| §10.2 alias/条件一段 | E5 |
| §10.5 首页表 | E12 · E13 · E27（三处插行，注意先后不影响正确性但影响可读顺序） |
| §11.3 `pre` 块 + bad 框 | E1（三处） · E4（两处） |
| §12.2 门表 | E11 · E29 |
| §12.3 | E8（三处） |
| §13.3 档表 + `span` | E6 · E7 · E14 |

## 5.4 与已落地修复的耦合（落盘方需一并回读的段落）

- **E1** ↔ L-029（§11.3 射程）· §10.2b `G-M2b` · §2.5 P5 —— 三处必须读成一套。
- **E5** ↔ L-008（§10.2 alias 来源④）· L-046（§0.1 falsification 挂区间）。
- **E7 / E20** ↔ L-011（`measured_quantity`，§3.1 与 §9.2）。
- **E16** ↔ L-018（§10.1 两种失效）· L-028（`G-X3` 凸包）。
- **E27** ↔ L-004（`own_coverage`，§10.4）· L-025（`U`，§14.2）。
- **E14 / E3** ↔ L-053（§13.3 bad 框）· L-076（`G-B4` 已移入 §2.8）。
