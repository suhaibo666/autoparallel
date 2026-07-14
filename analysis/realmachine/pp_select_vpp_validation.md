# pp × 选择重算 (× VPP) 真机验证 —— 2026-07-14 尝试记录

> **状态: BLOCKED —— 真机不可达（CorpLink VPN 未连），真机侧 0 配置跑成。**
> 本文档沉淀：① 阻塞证据；② 栈限制 ③（PP+重算互斥）的复核现状；③ 当前 HEAD 的仿真基线
> （恢复连通后直接填真机列算 ratio）；④ 精确到命令的恢复后执行清单。
> **本文不含任何新真机数据；表中"真机"列除既有锚点外全部待采。**

## 1. 连通性证据（2026-07-14 12:32–12:40，本 Windows 网关机）

- `ssh 192.168.9.116` 两次均失败：`kex_exchange_identification: Connection closed by remote host`。
- `fix-116-network` 诊断（`diagnose-116.ps1`）：L0/L1/L2 全 PASS（StarDream 7890 正常、星梦 MTU=1300
  已钳、ClaudeReverseTunnel/ClaudeVpnMtuClamp 任务 Ready），**L3 FAIL**：`server unreachable over
  VPN after 6 tries (lossy path / KEX stall)`。`repair-116-safe.ps1` 执行后复诊仍 `OVERALL: FAIL (vpn)`。
- **根因**：CorpLink VPN 客户端**进程未运行**（`Get-Process` 无 corplink/feilian 匹配），其 TAP 适配器
  `本地连接 (TAP-Windows Adapter V9 #2)` = Disconnected。到 116 的唯一路由是星梦（Meta Tunnel）的
  fake-IP 默认路由（接口 IP 198.18.0.1）；`ping 192.168.9.116` 通（TTL=128、<1ms）是**代理 TUN 假应答**，
  非真实连通。
- **处置**：需用户在本机手动连接 CorpLink（GUI+认证，无法远程自动化）后重试本清单 §4。
  这不属于 `fix-116-network` 可自修范围（该 skill 文档明确 "VPN down → the user must connect CorpLink first"）。

## 2. 栈限制 ③（PP+重算互斥）复核现状

- 记录于 2026-07-07（[[validation_matrix]] 栈限制 ③、参考文档 §15）：pp 配置只能 `NORECOMP=1`，
  P4 行 full/sel_attn/sel_ffn/sel_flash 全标 ✗。
- **本次未能取得任何新证据**（机器不可达）——该限制**既未被证实也未被证伪**，现状仍以 2026-07-07
  记录为准。注意：仓库内未找到当时 PP×select（区别于 PP×full）的原始报错栈 → §4 的最小配置探测
  （4L pp2 GBS2 select）**仍然必要**，其崩溃栈（若崩）本身就是交付物。
- VPP（`pipeline_parallel_interleave_num=2`）**从未真机验证**（07-09 报告 §6 明确列为无真机数据）。

## 3. 仿真基线（当前 HEAD `9f2372c`，恢复后直接对标）

**口径**：DSv3-mini 8L、seq4096、dp=1、pp=2、**B（每 microbatch）=2、m=2**、SP off、bf16/fp32、
`framework_reserve=0`。B=2 匹配既有 pp2 真机锚点（`cp2_none/DIAGNOSIS.md:30`：pp2-stage1 profiler
见 full-S **B2** 的 4040 MiB CE 缓冲）。select 经 `sel_cfg`（`self_attention:0-7` / `mlp:0-7`，
mindformers `select_module` 同口径）；vpp 经 `ParallelConfig.interleave`。

**调用路径一致性自检**：`serve_explorer.eval_config`（本表）与 `sim_vs_real_report.py` 直接 API 路径
在 HEAD 上对 pp2-none 逐字节一致（s0=10794.1 / s1=43899.1），锚点 ratio 复现 s0 **1.053** / s1 **0.962**。

| # | 配置 | sim s0 (MiB) | sim s1 (MiB) | 真机 s0 | 真机 s1 | ratio | 峰值事件 |
|---|---|---|---|---|---|---|---|
| e | pp2 · none · vpp1（锚点） | 10794.1 | 43899.1 | 10246✓ | 45655✓ | 1.053 / 0.962 | s0 bwd@4 / s1 bwd@9 |
| a | pp2 · select self_attention · vpp1 | 10311.0 | 26377.9 | 待采 | 待采 | — | s0 optstep / s1 bwd@9 |
| b | pp2 · select mlp · vpp1 | 10311.0 | 22066.6 | 待采 | 待采 | — | s0 optstep / s1 bwd@9 |
| c | pp2 · none · vpp2 | 10731.1 | 48634.6 | 待采 | 待采 | — | s0 bwd@2 / s1 bwd@9 |
| d | pp2 · select self_attention · vpp2 | 10311.0 | 29774.0 | 待采 | 待采 | — | s0 optstep / s1 bwd@9 |
| f | pp2 · select mlp · vpp2 | 10311.0 | 27618.3 | 待采 | 待采 | — | s0 optstep / s1 bwd@9 |

> [!warning] **VPP 行（c/d/f）为暂定值**：仿真器 VPP 的 m==pp 语义正在并行修复中
> （`mem_timeline.build_interleaved_1f1b`，m=2·pp=2·v=2 的 warmup 在飞深度口径），修复合入后
> **必须重跑本表 vpp 行**再对标。vpp2-none s1=48634（47.5 GiB）< `max_device_memory` 59GB，
> 理论可跑但余量小 → 真机 vpp 探测先用 LAYERS=4。
> 另：select 行 s0 峰移回 optstep（10311 < none 的 10794 bwd 峰）——重算削去逐层反向工作集所致，方向合理。

## 4. 恢复连通后的执行清单（按 skill §8，命令可直接粘贴）

```bash
REF=/home/suhaibo/workspace/mindformers/mindformers/tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3
# 0) 选空闲卡
ssh 192.168.9.116 "docker exec shb.ms.2.9 bash -lc 'npu-smi info | head -40'"
# 1) 栈探测（最小 4L,pp2+select;若崩,worker log 错误栈即交付物）
ssh 192.168.9.116 "docker exec shb.ms.2.9 bash -lc 'TAG=ppsel_probe LAYERS=4 DP=1 PP=2 GBS=2 \
  SELECT=self_attention:0-3 NORECOMP=0 PORT=8155 CARDS=<i,j> W=2 bash $REF/run_axis.sh'"
# 2) 跑通则升 8L: (a) SELECT=self_attention:0-7 (b) SELECT=mlp:0-7（各 3 步）
# 3) vpp 探测: 先 LAYERS=4 NORECOMP=1 + interleave=2。⚠ run_axis.sh 的 env 清单里无 INTERLEAVE——
#    先 cat $REF/run_axis.sh 确认是否透传;若无,export SIM_INTERLEAVE=2 直调 prep_ds3_axis.py+msrun
#    （prep 支持 SIM_INTERLEAVE→pipeline_parallel_interleave_num,本仓 .claude/skills/.../prep_ds3_axis.py:25,63）。
# 4) vpp2+select（仅当 1)和 3)都活着）
# 5) 采数: ssh ... "grep -h MEMPROBE $REF/log_<TAG>/worker_*.log"  → 填 §3 表、算 ratio
# 6) select 生效核验: worker log 搜 'Final Select Recompute Configuration Map'（防静默不重算假数据）
# 7) 礼仪: 只占 2 卡、3 步、跑完删 log_<TAG>
```

真机原始日志路径（跑成后回填）：`$REF/log_<TAG>/worker_*.log`（server 侧）。

## 关联

- [[validation_matrix]]（P4 行与栈限制 ③ 的原始记录）
- `analysis/realmachine/sim_vs_real_report_2026-07-09.md`（12 锚点基线、pp2 锚点出处）
- `analysis/realmachine/cp2_none/DIAGNOSIS.md`（k_ce=8 标定自 pp2-stage1、B2 口径证据）
- `specs/2026-07-07-memory-model-reference.md` §15（栈限制与诚实边界）
- `.claude/skills/real-machine-memory-sim/SKILL.md` §8（run_axis.sh 配方）
