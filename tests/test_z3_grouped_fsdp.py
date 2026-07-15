"""Z3：grouped-FSDP 子域建模（把上一轮的 fail-loud 变真建模）。

真机语义（`../mindformers/mindformers/pynative/distributed/parallel_dims.py:443-470`
`get_fsdp_shard_mesh`）：完整 FSDP 域 `fsdp = dp_shard·cp`；`dense_fsdp_shard_size` 是**子域**大小
（须整除 fsdp、∈[1,fsdp]）。dense（非专家）权重按**子域**分片：每卡 dense 持久 =
`dense_global / dense_fsdp_shard_size`（而非 /fsdp）；子域外的 dp 维对 dense 是**复制** → 每卡驻留
**更大**（÷更小域）。expert 权重走独立 efsdp，不受 dense_fsdp_shard_size 影响。

上一轮（closure-w1 F1）对 `1<=v<fsdp` fail-loud；本轮真建模——映射到
`ParallelConfig.dense_fsdp_shard_size` 并让 static_mem 用子域分母切 dense 持久态。
"""
import pytest

from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.layers.moe import build_moe_decoder
from cost_eval.specs import ParallelConfig, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.configs.from_mindformers import from_mindformers_dict


# ── 纯 dense toy（与 test_static_mem 同 D；每层 dense=656 numel，全部 ÷8/÷2）──────────────
D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
_DENSE_PER_LAYER = 656          # ln1_g8+qkv192+o64+ln2_g8+fc1_256+fc2_128（全部 ÷8）
_STATE = 14                     # AdamW bf16 param+opt（剔 grad）


def _dense_spec():
    return ModelSpec("toy", D, ["dense", "dense"], {"dense": build_dense_decoder(D)})


def _persistent(pc):
    pm = ParallelModel(pc, n_layers=2, world_size=64)
    g = ShapeEval().resolve(_dense_spec(), pm)
    return StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)[0]


# ══ ParallelConfig 字段 & 校验 ═══════════════════════════════════════════════════════════
def test_field_default_is_zero_neutral():
    """缺省 dense_fsdp_shard_size==0（惰性=用完整 fsdp）。"""
    assert ParallelConfig().dense_fsdp_shard_size == 0


def test_field_accepts_divisor_leq_fsdp():
    """子域须整除完整 fsdp=dp_shard·cp 且 ≤fsdp：8/2/4 合法，==fsdp(8) 合法。"""
    for v in (1, 2, 4, 8):
        assert ParallelConfig(dp_shard=8, dense_fsdp_shard_size=v).dense_fsdp_shard_size == v


def test_field_rejects_non_divisor():
    """shard=3 不整除 fsdp=8 → ValueError（仿 parallel_dims.py:464-468）。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        ParallelConfig(dp_shard=8, dense_fsdp_shard_size=3)


def test_field_rejects_over_fsdp():
    """shard=16 > fsdp=8 → ValueError。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        ParallelConfig(dp_shard=8, dense_fsdp_shard_size=16)


def test_field_rejects_bool():
    """bool 是 int 子类,须先排除（仿 W2/F5a 整数守卫、parallel_dims.py:458-462）。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        ParallelConfig(dp_shard=8, dense_fsdp_shard_size=True)


def test_field_rejects_negative():
    """shard=-1 非正 → ValueError。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        ParallelConfig(dp_shard=8, dense_fsdp_shard_size=-1)


def test_field_domain_includes_cp():
    """完整 fsdp 含 cp：dp_shard=2·cp=2→fsdp=4；shard=4/2 整除合法、shard=3 拒。"""
    assert ParallelConfig(dp_shard=2, cp=2, dense_fsdp_shard_size=2).dense_fsdp_shard_size == 2
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        ParallelConfig(dp_shard=2, cp=2, dense_fsdp_shard_size=3)


# ══ ParallelModel.dense_fsdp_degree() helper ═════════════════════════════════════════════
def test_dense_fsdp_degree_default_is_full_fsdp():
    """未配 → dense_fsdp_degree() == fsdp_degree()（= dp_shard·cp）。"""
    pm = ParallelModel(ParallelConfig(dp_shard=4, cp=2), n_layers=2, world_size=8)
    assert pm.dense_fsdp_degree() == pm.fsdp_degree() == 8


def test_dense_fsdp_degree_subdomain():
    """配子域 → dense_fsdp_degree() == 子域大小（< 完整 fsdp）。"""
    pm = ParallelModel(ParallelConfig(dp_shard=8, dense_fsdp_shard_size=2), n_layers=2, world_size=8)
    assert pm.dense_fsdp_degree() == 2
    assert pm.fsdp_degree() == 8          # 完整域 helper 不受影响


# ══ static_mem：子域 dense 持久 = global/shard > global/fsdp ══════════════════════════════
def test_default_persistent_byte_identical_to_full_fsdp():
    """缺省(0/未设)→ 逐字节 == 完整 fsdp 分片。dp_shard=8 → 每卡 dense 持久 = 656·2/8·14。"""
    expect = (_DENSE_PER_LAYER // 8) * 2 * _STATE          # 82·2·14 = 2296
    assert _persistent(ParallelConfig(dp_shard=8)) == expect
    # 显式 dense_fsdp_shard_size=0 与未设同值（惰性）
    assert _persistent(ParallelConfig(dp_shard=8, dense_fsdp_shard_size=0)) == expect


def test_subdomain_dense_persistent_is_global_over_shard():
    """子域 shard=2 → 每卡 dense 持久 = global/2（> global/fsdp=global/8）。"""
    full = _persistent(ParallelConfig(dp_shard=8))                       # ÷8
    sub = _persistent(ParallelConfig(dp_shard=8, dense_fsdp_shard_size=2))  # ÷2
    assert sub == (_DENSE_PER_LAYER // 2) * 2 * _STATE                   # 328·2·14 = 9184
    assert sub > full


def test_subdomain_persistent_4x_example():
    """数值例子（任务指定）：fsdp=8、shard_size=2 → dense 持久 ×4（÷2 vs ÷8）。"""
    full = _persistent(ParallelConfig(dp_shard=8))
    sub = _persistent(ParallelConfig(dp_shard=8, dense_fsdp_shard_size=2))
    assert sub == 4 * full


def test_shard_equals_fsdp_is_full_domain():
    """shard==fsdp(8) → 与完整域逐字节一致（复用完整 FSDP mesh 语义）。"""
    assert (_persistent(ParallelConfig(dp_shard=8, dense_fsdp_shard_size=8))
            == _persistent(ParallelConfig(dp_shard=8)))


# ══ experts 不受影响（走独立 efsdp）══════════════════════════════════════════════════════
_MD = DimTable(H=16, F=32, n_heads=4, n_kv=4, head_dim=4, S=8, B=1, vocab=16, n_layers=1,
               n_experts=8, topk=2, moe_F=32)


def _moe_spec():
    return ModelSpec("moe", _MD, ["moe"], {"moe": build_moe_decoder(_MD)})


def _dense_global_numel(g):
    """resolved graph 里**非专家**权重 numel 之和（按名去重/层）——dense 子域分片的分子。"""
    total = 0
    for layers in g.stages.values():
        for layer in layers:
            seen = {}
            for op in layer.ops:
                for w in op.params:
                    if not getattr(w, "is_expert", False):
                        seen[w.name] = w.local_numel
            total += sum(seen.values())
    return total


def test_experts_unaffected_only_dense_scales():
    """MoE 层：配 dense 子域后，总持久增量 == 仅 dense 缩放（experts 走 efsdp,不变）。

    ep=1 → efsdp=8；dense_fsdp: 8(full) vs 2(sub)。delta = dense_global·(1/2−1/8)·14。
    若 experts 被误缩放,delta 不会等于纯 dense 公式。"""
    pm_full = ParallelModel(ParallelConfig(dp_shard=8, ep=1), n_layers=1, world_size=8)
    pm_sub = ParallelModel(ParallelConfig(dp_shard=8, ep=1, dense_fsdp_shard_size=2),
                           n_layers=1, world_size=8)
    g_full = ShapeEval().resolve(_moe_spec(), pm_full)
    g_sub = ShapeEval().resolve(_moe_spec(), pm_sub)
    p_full = StaticMem().compute(g_full, OptimizerSpec.adamw(), pm_full, cpu_offload=False)[0]
    p_sub = StaticMem().compute(g_sub, OptimizerSpec.adamw(), pm_sub, cpu_offload=False)[0]

    dense_g = _dense_global_numel(g_full)
    expect_delta = ((dense_g // 2) - (dense_g // 8)) * _STATE
    assert p_sub - p_full == expect_delta
    assert expect_delta > 0                    # dense 确实变大


# ══ adapter：from_mindformers 映射 1<=v<fsdp 建模、==fsdp 中性、非法拒 ═════════════════════
def _mf(parallelism):
    return {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                      "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"},
            "training": {"local_batch_size": 1}, "parallelism": parallelism}


def test_adapter_maps_subdomain_to_config():
    """1<=v<fsdp 且整除（dp_shard=8,shard=2）→ 映射到 ParallelConfig.dense_fsdp_shard_size=2（建模,不再 fail-loud）。"""
    bundle = from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 2}))
    assert bundle.parallel.dense_fsdp_shard_size == 2


def test_adapter_maps_shard_one():
    """shard=1（极端子域,dense 全复制）→ 映射为 1（建模）。"""
    bundle = from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 1}))
    assert bundle.parallel.dense_fsdp_shard_size == 1


def test_adapter_equals_fsdp_is_neutral_zero():
    """shard==fsdp(8) → 复用完整 mesh,中性 → dense_fsdp_shard_size=0（不设）。"""
    bundle = from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 8}))
    assert bundle.parallel.dense_fsdp_shard_size == 0


def test_adapter_absent_is_zero():
    """未配 → 0（完整 fsdp,惰性）。"""
    bundle = from_mindformers_dict(_mf({"data_parallel_shard": 8}))
    assert bundle.parallel.dense_fsdp_shard_size == 0


def test_adapter_rejects_non_divisor():
    """shard=3 不整除 fsdp=8 → ValueError（runtime 约束,parallel_dims.py:464-468）。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 3}))


def test_adapter_rejects_over_fsdp():
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 16}))


def test_adapter_rejects_zero():
    """value=0 显式配 → 非正,拒（parallel_dims.py:458-462）。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf({"data_parallel_shard": 8, "dense_fsdp_shard_size": 0}))


def test_adapter_subdomain_below_fsdp_with_cp():
    """完整域含 cp：dp_shard=2·cp=2→fsdp=4；shard=2<4 整除 → 建模映射为 2。"""
    bundle = from_mindformers_dict(_mf({
        "data_parallel_shard": 2, "context_parallel": 2, "dense_fsdp_shard_size": 2}))
    assert bundle.parallel.dense_fsdp_shard_size == 2
    assert bundle.parallel.cp == 2 and bundle.parallel.dp_shard == 2
