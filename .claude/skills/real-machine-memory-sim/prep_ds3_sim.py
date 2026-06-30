"""构造缩层 DSv3 pynative 仿真 config + 生成小合成数据集。

在参考测试目录下运行（含 pynarive_ds3.yaml），且 PYTHONPATH 含 mindformers 仓库根（为 tests.utils）：
    SIM_LAYERS=4 SIM_STEPS=3 python prep_ds3_sim.py
产出：同目录 ds3_sim.yaml + train_dataset_sim/ 合成数据。参考 test_two_cards.py 的 generate_dataset。
"""
import os
import yaml

CUR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(CUR, "pynarive_ds3.yaml")
DS_DIR = os.path.join(CUR, "train_dataset_sim")
DS_FILE = os.path.join(DS_DIR, "dataset.mindrecord")

N_LAYERS = int(os.environ.get("SIM_LAYERS", "4"))
STEPS = int(os.environ.get("SIM_STEPS", "3"))
SEQ = int(os.environ.get("SIM_SEQ", "4096"))
EP = int(os.environ.get("SIM_EP", "1"))         # 专家并行度（变配置验证用）
TP = int(os.environ.get("SIM_TP", "1"))         # 张量并行度
PP = int(os.environ.get("SIM_PP", "1"))         # 流水并行度
DPSHARD = int(os.environ.get("SIM_DPSHARD", "-1"))  # FSDP shard 度（-1=auto）

# 1) 合成数据集（input_ids/labels/loss_mask/position_ids），仅几十条
if not os.path.exists(DS_FILE):
    from tests.utils.generate_dataset import generate_mindrecord_file
    os.makedirs(DS_DIR, exist_ok=True)
    generate_mindrecord_file(
        seq_length=SEQ, batch_size=2, train_steps=20, dataset_path=DS_FILE,
        data_schema={
            "input_ids": {"type": "int32", "shape": [-1]},
            "labels": {"type": "int32", "shape": [-1]},
            "loss_mask": {"type": "int32", "shape": [-1]},
            "position_ids": {"type": "int32", "shape": [-1]},
        },
    )
    print("DATASET_GENERATED", DS_DIR)
else:
    print("DATASET_EXISTS", DS_DIR)

# 2) 缩层 config
with open(BASE) as f:
    cfg = yaml.safe_load(f)
cfg["train_dataset"]["dataloader"]["dataset_files"] = [DS_DIR]
cfg["training"]["local_batch_size"] = 1
cfg["training"]["global_batch_size"] = 2          # 2 cards × 1（FSDP-only）
cfg["training"]["steps"] = STEPS
cfg["model"]["num_hidden_layers"] = N_LAYERS
cfg["model"]["seq_length"] = SEQ
cfg["recompute"]["full_recompute_layer"] = [f"0-{N_LAYERS - 1}"]
cfg["checkpoint"]["enable_save"] = False
cfg["parallelism"]["expert_parallel"] = EP      # 变配置：EP
cfg["parallelism"]["tensor_parallel"] = TP      # 变配置：TP
cfg["parallelism"]["pipeline_parallel"] = PP    # 变配置：PP
cfg["parallelism"]["data_parallel_shard"] = DPSHARD
if PP > 1:
    cfg["parallelism"]["pipeline_parallel_microbatch_size"] = PP   # 至少 PP 个 microbatch

out = os.path.join(CUR, "ds3_sim.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f, indent=2)
print("WROTE_CONFIG", out, "layers", N_LAYERS, "steps", STEPS, "seq", SEQ)
