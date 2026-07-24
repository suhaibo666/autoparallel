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


class FrameworkGapWarning(UserWarning):
    """**框架缺口**咨询（2026-07-24 口径切换）：评估器按 MindSpore/mindformers 真实代码语义做
    **纯理论**估计，不做任何经验补偿；理论 vs 真机的差距作为框架缺口**显式暴露**、不吸收进数字。

    典型场景：full-recompute 下评估器按 MS checkpoint 理论语义只留每微批层的重算边界（~128MiB
    层入口），但 MS2.10 真机每微批层实际驻留约 1.9GB、全重算只释放约 30% 激活——差距是**框架释放
    缺口**，非模型/建模误差。**这类警示务必让用户看见：理论峰值显著低于真机实测，OOM 判断勿直接
    采用此理论值。**"""


def warn_oom_safety(msg: str, *, stacklevel: int = 2) -> None:
    """发一条 OOM-安全咨询(欠预测风险)。薄封装,便于统一 stacklevel 与测试断言。"""
    warnings.warn(msg, OOMSafetyWarning, stacklevel=stacklevel + 1)


def warn_framework_gap(msg: str, *, stacklevel: int = 2) -> None:
    """发一条框架缺口咨询（纯理论 vs 真机差距，显式暴露、不进数字）。"""
    warnings.warn(msg, FrameworkGapWarning, stacklevel=stacklevel + 1)


def warn_modeling_approx(msg: str, *, stacklevel: int = 2) -> None:
    """发一条建模近似 / 内存中性未建模字段咨询。"""
    warnings.warn(msg, ModelingApproxWarning, stacklevel=stacklevel + 1)
