"""配置文件转换器（D-7）：外部框架 config → 评估器配置对象。

- `from_mindformers`：mindformers pynative 训练 yaml（dict 或路径）→ `EvaluatorConfigBundle`
  （`LLMConfig` + `ParallelConfig`/`RecomputeSpec`/`SwapSpec`/`OptimizerSpec`/`HardwareSpec`）。

**纯核**（`from_mindformers_dict`）不 import yaml；仅路径壳 `load_mindformers_yaml` 内惰性导入，
故 `cost_eval` 核不新增硬 yaml 依赖。
"""
from .from_mindformers import (
    EvaluatorConfigBundle,
    from_mindformers_dict,
    load_mindformers_yaml,
)

__all__ = [
    "EvaluatorConfigBundle",
    "from_mindformers_dict",
    "load_mindformers_yaml",
]
