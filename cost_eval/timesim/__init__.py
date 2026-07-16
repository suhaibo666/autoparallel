# cost_eval/timesim/__init__.py
"""timesim：step time 仿真器（spec specs/2026-07-16-step-time-cost-model-design.md）。

与内存仿真器（mem_timeline/structure_mem/static_mem）**双向 import 禁令**（契约 §2.2-1，
tests/test_timesim_decoupling.py 强制）。允许共享：cost_eval.specs / cost_eval.schedule /
cost_eval.opdag（上游 IR 资产）。"""
