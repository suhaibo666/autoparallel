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
    """**理论口径 vs 真机残余偏差**咨询（2026-07-25 重写；类名保留兼容 import/filter）：评估器按
    MindSpore/mindformers 真实代码语义做**纯理论**估计，不做任何经验补偿；理论 vs 真机的残余差
    **显式暴露**、不吸收进数字。

    典型场景：full-recompute。**旧措辞「框架不释放 / 全重算仅释放约 30% 激活」已被实测证伪**——
    167/MS2.10 单卡微基准（2026-07-25，`log_release_probe/VERDICT_2026-07-25.txt`）测得 rc=ON
    前向末残留 **≡ 0**（NBLK=2/4/8，bare_ctx/saved/pyref 三锚同值）、rc=ON 前向峰**与微批深度
    无关**（复刻 552 / 真模块 403 恒定）→ 重算**确实**释放激活，残余量是**单区域瞬态 ×1**。
    真正的缺口是**重算工作集未建模**：重算再执行必须把 backward 需要的整个 saved 集同时物化并
    活到 backward 消费完（`remat_saves = activation_saves − checkpoint_input`，2026-07-25 补建）。

    **补建后残余偏差仍在，方向依配置而异**：站点 pp8 全重算理论约为真机 74%（**欠读**）；1 层/
    stage 的极端 PP 配置上因 `recomp_scratch`/`bwd_working_set`/`remat_saves` 都从
    `forward_max_live`/saves 派生而**部分重叠**，会**过读** 1.1~1.3×。**这类警示务必让用户看见：
    OOM 判断勿直接采用理论值（欠读侧会误报"放得下"）。**"""


def warn_oom_safety(msg: str, *, stacklevel: int = 2) -> None:
    """发一条 OOM-安全咨询(欠预测风险)。薄封装,便于统一 stacklevel 与测试断言。"""
    warnings.warn(msg, OOMSafetyWarning, stacklevel=stacklevel + 1)


def warn_framework_gap(msg: str, *, stacklevel: int = 2) -> None:
    """发一条理论口径 vs 真机残余偏差咨询（显式暴露、不进数字；见 `FrameworkGapWarning`——
    「框架不释放」旧理论已被 167/MS2.10 微基准证伪，现指重算工作集口径的残余偏差）。"""
    warnings.warn(msg, FrameworkGapWarning, stacklevel=stacklevel + 1)


def warn_modeling_approx(msg: str, *, stacklevel: int = 2) -> None:
    """发一条建模近似 / 内存中性未建模字段咨询。"""
    warnings.warn(msg, ModelingApproxWarning, stacklevel=stacklevel + 1)
