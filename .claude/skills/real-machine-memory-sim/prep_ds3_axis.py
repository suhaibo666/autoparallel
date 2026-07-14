"""构造缩层 DSv3 pynative 仿真 config（**多轴** 版：在 prep_ds3_sim.py 基础上加 CP / SP /
global_batch / microbatch_num / interleave 控制），供并行轴单变量真机验证。

在参考测试目录下运行（含 pynarive_ds3.yaml），PYTHONPATH 含 mindformers 仓库根：
    SIM_LAYERS=8 SIM_PP=2 python prep_ds3_axis.py
产出：同目录 ds3_sim.yaml + train_dataset_sim/ 合成数据。
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
EP = int(os.environ.get("SIM_EP", "1"))
TP = int(os.environ.get("SIM_TP", "1"))
PP = int(os.environ.get("SIM_PP", "1"))
CP = int(os.environ.get("SIM_CP", "1"))
DPSHARD = int(os.environ.get("SIM_DPSHARD", "-1"))
GBS = int(os.environ.get("SIM_GBS", "2"))
INTERLEAVE = int(os.environ.get("SIM_INTERLEAVE", "1"))
# microbatch 数：默认 PP（PP>1 时至少 PP 个），可用 SIM_MBN 覆盖
MBN = int(os.environ.get("SIM_MBN", str(PP if PP > 1 else 1)))
SP = os.environ.get("SIM_SP", "")   # "true"/"false"/"" (=保持 base)

# 1) 合成数据集（复用；schema 同 test_two_cards）
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
cfg["training"]["global_batch_size"] = GBS
cfg["training"]["steps"] = STEPS
cfg["model"]["num_hidden_layers"] = N_LAYERS
cfg["model"]["seq_length"] = SEQ
cfg["recompute"]["full_recompute_layer"] = [f"0-{N_LAYERS - 1}"]
cfg["checkpoint"]["enable_save"] = False
cfg["parallelism"]["expert_parallel"] = EP
cfg["parallelism"]["tensor_parallel"] = TP
cfg["parallelism"]["pipeline_parallel"] = PP
cfg["parallelism"]["context_parallel"] = CP
cfg["parallelism"]["data_parallel_shard"] = DPSHARD
cfg["parallelism"]["pipeline_parallel_interleave_num"] = INTERLEAVE
if PP > 1:
    cfg["parallelism"]["pipeline_parallel_microbatch_size"] = MBN
if SP in ("true", "false"):
    cfg["parallelism"]["sequence_parallel"] = (SP == "true")

out = os.path.join(CUR, "ds3_sim.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f, indent=2)
print("WROTE_CONFIG", out, "layers", N_LAYERS, "TP", TP, "PP", PP, "CP", CP,
      "EP", EP, "DPSHARD", DPSHARD, "GBS", GBS, "MBN", MBN, "INTERLEAVE", INTERLEAVE)
