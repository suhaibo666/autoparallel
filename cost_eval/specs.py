"""并行/优化器/硬件/重算/swap 配置。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


_CP_METHODS = ("colossal", "ulysses", "ring", "hybrid")


@dataclass
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    # context-parallel 算法（`parallelism.context_parallel_method`，忠实 mindformers
    # pynative/distributed/{context_parallel,style}.py）。默认 `colossal`（对齐 mindformers DSv3
    # yaml）。决定 body 激活的 cp 切分口径（D-1 修正 2026-07-07，真机确认）：
    #   - ulysses/ring/hybrid：body 激活（含 attention KV）÷cp。
    #   - colossal（ulysses_degree=1）：body ÷cp，但 **attention KV all-gather 到 full-S**（额外 KV buffer）。
    # loss/head 区亦随 cp **÷cp**（P2-08 对齐 2026-07-14：早期"full-S 实测"是把 B=2·S/cp 误读为
    # B=1·full-S——Bug A，cp2-none profiler 复核证 loss buffer=[S/cp,B,V]，权威口径见 head.py:59-64）。
    # cp=1 时该字段无效（cp 分支不进入）→ DSv3 anchor 逐字节不变。
    context_parallel_method: str = "colossal"
    reshard_after_forward: str = "default"   # always|never|default
    cpu_offload: bool = False
    microbatch: int = 1
    interleave: int = 1
    # PP 每 stage 层数（含 embedding+head 两伪层，和==n_layers）：**首选显式配置**（忠实
    # mindformers `offset`/`num_layer_list`，D-8），不自行推测；None=退化均匀切（remainder 归末 stage）。
    layers_per_stage: Optional[list] = None
    prefetch_depth: int = 1
    num_microbatches: int = 1                # m = global_batch/(dp*microbatch)，由 adapter 算好
    # grouped-FSDP 子域大小（Z3，2026-07-15，忠实 mindformers `pynative/distributed/parallel_dims.py:443-470`
    # `get_fsdp_shard_mesh`）：dense（非专家）权重在**子域**（size = dense_fsdp_shard_size）上分片,
    # 而非完整 FSDP 域 `fsdp = dp_shard·cp`。子域外的 dp 维对 dense 是**复制** → 每卡 dense 持久 =
    # `dense_global / dense_fsdp_shard_size`（÷更小域 → 更大）。expert 权重走独立 efsdp,**不受此字段影响**。
    # **0/None = 惰性**（用完整 fsdp）→ 现有 spec/锚点逐字节不变。>0 时须整除 fsdp 且 ∈[1,fsdp]。
    dense_fsdp_shard_size: int = 0

    def __post_init__(self):
        # cp 算法 fail-loud（仿 head.py:50 loss_type / build_llm._check_implemented_dispatch）：
        # 非实现取值即报错，不静默按某算法继续（会产「貌似合理实则错误」的切分）。
        if self.context_parallel_method not in _CP_METHODS:
            raise ValueError(
                f"context_parallel_method={self.context_parallel_method!r} 未实现"
                f"（支持：{_CP_METHODS}）")
        # closure-audit C1（2026-07-15）：reshard 枚举是**本字段自身不变量** → 在核心构造处校验，
        # 不只靠 adapter 封口（此前 typo `nevver` 静默回落到 default，与配置意图相反）。
        if self.reshard_after_forward not in ("always", "never", "default"):
            raise ValueError(
                f"reshard_after_forward={self.reshard_after_forward!r} 非法"
                "（仅 always|never|default）——typo 静默回落会评错 gather 生命周期。")
        # closure-audit P1-16（2026-07-15）：并行/调度标量正值校验——同属本字段自身不变量，在核心
        # 构造处 fail-loud（此前 interleave=0 / prefetch_depth=-1 / num_microbatches=0 被接受，
        # 令调度退化、空循环或产无意义峰值）。默认构造（全 1）与所有合法值必须继续通过。
        _at_least_1 = {
            "dp_replicate": self.dp_replicate, "dp_shard": self.dp_shard,
            "cp": self.cp, "tp": self.tp, "pp": self.pp, "ep": self.ep,
            "interleave": self.interleave, "microbatch": self.microbatch,
            "num_microbatches": self.num_microbatches,
        }
        # closure-audit v2 §F5a（2026-07-15）：**排除 bool**。Python `bool` 是 `int` 子类
        # （`True==1`/`False==0`），故 `isinstance(True,int)` 为真且 `True>=1` 通过——旧校验
        # 会把 `tp=True`/`num_microbatches=True` 当整数并行度接受。判据须**严格整数**：先拒 bool，
        # 再拒非 int，再拒 <1。报错含字段名/值/类型。
        for _name, _val in _at_least_1.items():
            if isinstance(_val, bool) or not isinstance(_val, int) or _val < 1:
                raise ValueError(
                    f"ParallelConfig.{_name}={_val!r}（类型 {type(_val).__name__}）非法——须为 >=1 "
                    "的严格整数（bool 不算整数；并行度/VPP 交错/microbatch 至少 1，<1 会令调度退化/"
                    "空循环/无意义峰）。")
        if (isinstance(self.prefetch_depth, bool) or not isinstance(self.prefetch_depth, int)
                or self.prefetch_depth < 0):
            raise ValueError(
                f"ParallelConfig.prefetch_depth={self.prefetch_depth!r}"
                f"（类型 {type(self.prefetch_depth).__name__}）非法——须为 >=0 的严格整数"
                "（bool 不算整数；0=无预取合法，负数无意义）。")
        # closure-audit Z3（2026-07-15）：grouped-FSDP 子域自身不变量——>0 时须**严格正整数**、
        # 整除完整 fsdp=dp_shard·cp 且 ∈[1,fsdp]（仿上方 F5a 整数守卫 + parallel_dims.py:458-468：
        # 先拒 bool 再拒非 int 再查整除/范围；bool 是 int 子类会溜过 `fsdp % True == 0`）。0/None=惰性,
        # 跳过（缺省逐字节不变）。**只在核心构造处校验**——不整除会令 dense 子域切分低估每卡持久→OOM 不安全。
        if self.dense_fsdp_shard_size not in (None, 0):
            fsdp = self.dp_shard * self.cp
            v = self.dense_fsdp_shard_size
            if (isinstance(v, bool) or not isinstance(v, int)
                    or v < 1 or v > fsdp or fsdp % v != 0):
                raise ValueError(
                    f"ParallelConfig.dense_fsdp_shard_size={v!r}"
                    f"（类型 {type(v).__name__}）非法——须为 >=1 的严格整数、∈[1,{fsdp}] 且整除完整 "
                    f"fsdp=dp_shard·cp={fsdp}（bool 不算整数；0/None=惰性用完整 fsdp）。")


@dataclass
class OptimizerSpec:
    type: str = "AdamW"
    # 持久 = param + optimizer state（**剔除 grad**，真机修正 §8.4）
    state_bytes_per_param: int = 14          # bf16 params: bf16 2 + fp32 master4 + m4 + v4
    grad_dtype_bytes: int = 4                # 反向瞬态 grad(grad_buf) 的 dtype：fp32=4 / bf16=2

    @classmethod
    def adamw(cls, params_fp32: bool = False, grad_dtype_bytes: int = 4) -> "OptimizerSpec":
        # 持久(剔grad): fp32 params=master4+m4+v4=12; bf16 params=bf16 2+master4+m4+v4=14
        return cls("AdamW", 12 if params_fp32 else 14, grad_dtype_bytes)


@dataclass
class HardwareSpec:
    max_device_memory: int                   # bytes（来自 ContextConfig.max_device_memory）
    # framework_reserve：**审计/回归旋钮**（默认 0，非生产项）。生产路径下框架瞬态已按机理
    # 拆进 op 图（FSDP 预取→gather_buf、flash-ws→flash workspace、MoE staging→dispatch/combine
    # workspace，见 framework.py）；此项仅在显式给值时复现旧「经验兜底常数」以供审计（如 golden
    # 的 177 MiB）。**不是拟合的物理量**。
    framework_reserve: int = 0
    # alloc_block_bytes：**平台属性**（不是拟合值）——MindSpore 设备内存池 `DynamicMemPoolBestFit`
    # 对每次分配按 `kMemAlignSize`(=512B, `kDynamicMemAlignSize`) 对齐。分配峰值(max_memory_allocated)
    # 里每个张量按此块粒度上取整 → 少量对齐碎片。这是 framework_reserve「分配器块对齐」项的**公式化**
    # 落地（structure_mem 逐张量 roundup），取代经验常数。默认 512（可按硬件覆盖）。
    alloc_block_bytes: int = 512


@dataclass
class RecomputeSpec:
    """重算配置。三种 mode（忠实 mindformers `RecomputeConfig.mode`，config.py:759-796）：

    - ``None``：不重算，全量 saves 常驻。
    - ``full``：`full_layers` 里的层整层重算（仅保留层入口 checkpoint_input）。
    - ``select``：**选择性重算**——每层只重算选中的一组 op/模块，其 saves 丢弃、反向重物化，
      未选中 op 的 saves 仍常驻（内存↔重算细粒度旋钮）。

    `select_ops` 是 `{layer_id -> set[str]}`：每层一组**选择器**（op 名或 op 类型的子串，
    大小写不敏感）。这对应 mindformers `select_module`（`{module_path: [layer_ranges]}`）经
    `_clean_and_parse_config`（activation_checkpoint.py:478-518）**反转**后的 `{layer_id:
    [module_names]}`；本库 op 图是扁平叶子模块，故「模块」= 一组 op，用**子串命中**匹配
    （忠实 mindformers `exclude_op` 的「算子名含子串即命中」语义，config.py:779-789）。
    选中判定同时看 op 名与 op 类型 → `{"flash"}` / `{"attn"}` 命中 flash_attn（Megatron
    默认 selective = core_attn，transformer_config.py:526-534/559）。
    """
    mode: str = "None"                       # None|full|select
    full_layers: set = field(default_factory=set)
    select_ops: dict = field(default_factory=dict)   # layer_id -> set[str]（选中的 op 名/类型子串）

    def is_full(self, layer_id: int) -> bool:
        return self.mode == "full" and layer_id in self.full_layers

    def is_select(self, layer_id: int) -> bool:
        """该层是否选择性重算：mode==select 且配了**非空**选择器集。"""
        return self.mode == "select" and bool(self.select_ops.get(layer_id))

    def selectors(self, layer_id: int) -> set:
        """该层的选择器集（非 select 模式恒空集；未配层返回空集）。"""
        if self.mode != "select":
            return set()
        return set(self.select_ops.get(layer_id, set()))

    def op_matches(self, layer_id: int, op_name: str, op_type: str) -> bool:
        """某 op 是否被该层选中重算：任一选择器是 op 名或 op 类型的子串（大小写不敏感）。

        忠实 mindformers：`select_module` 命模块路径（此处退化为 op 名子串）、`exclude_op`
        命 op 名子串（activation_checkpoint.py:534-540 `needle in op_name`）；同时兼容按
        op 类型选（如 `"flash_attn"` / `"attn"` → Megatron core_attn）。"""
        name = (op_name or "").lower()
        typ = str(op_type or "").lower()
        return any(s.lower() in name or s.lower() in typ
                   for s in self.selectors(layer_id))


@dataclass
class SwapSpec:
    enable: bool = False
    default_prefetch: int = 1
    swap_layers: set = field(default_factory=set)

    def swaps(self, layer_id: int) -> bool:
        return self.enable and layer_id in self.swap_layers
