"""建模咨询告警（round3 A 档，2026-07-16）——集中定义**非致命** advisory 类别,供 build_llm /
parallel_model / report / mem_timeline 等处发出结构化提示。不改任何数值、不产错数,只提示用户。

分两类(语义不同,便于用户按需 filter/elevate):
  - `ModelingApproxWarning`:未建模的**内存中性**字段、或**建模近似**(方向中性或已论证不上峰)。
    例:add_bias_linear(bias 内存可忽略)、VPP `layers_per_stage`+interleave 连续切近似。
  - `OOMSafetyWarning`:预估可能**欠预测**(OOM-**不安全**方向)的咨询——提示用户留安全余量。
    例:缩层锚点外推全尺寸的累计每层残差、无重算-MoE margin 未开(直连 API 默认 0)。
    **这类务必让用户看见**:欠预测会误报"放得下"却 OOM,是最危险的方向。

用法:`from .advisories import OOMSafetyWarning; warnings.warn(msg, OOMSafetyWarning, stacklevel=2)`。
默认 Python warning filter 对同一 (message, category, module, lineno) 只显示一次,故循环内发也不刷屏。
"""
import warnings


class ModelingApproxWarning(UserWarning):
    """未建模的内存中性字段 / 建模近似(方向中性)——不产错数,只提示。"""


class OOMSafetyWarning(UserWarning):
    """预估可能欠预测(OOM-不安全方向)的咨询——提示留安全余量,不改数值。"""


def warn_oom_safety(msg: str, *, stacklevel: int = 2) -> None:
    """发一条 OOM-安全咨询(欠预测风险)。薄封装,便于统一 stacklevel 与测试断言。"""
    warnings.warn(msg, OOMSafetyWarning, stacklevel=stacklevel + 1)


def warn_modeling_approx(msg: str, *, stacklevel: int = 2) -> None:
    """发一条建模近似 / 内存中性未建模字段咨询。"""
    warnings.warn(msg, ModelingApproxWarning, stacklevel=stacklevel + 1)
