"""开发/验收工具包（脚本亦可直接 `python tools/<name>.py` 跑）。

  * `liveness_ab_validate` —— 167 A/B 八跑**验收门**（source × config，见该模块 docstring）。
  * `audit_saves`          —— 手写 `saves` 名册 vs 源真值 `save_for_backward` 的审计台账。

做成包是为了让 `tests/` 能 `from tools.liveness_ab_validate import ...` 复用验收门的逻辑
（真机地面真值表、八跑派生、不变量），避免测试里再抄一遍。
"""
