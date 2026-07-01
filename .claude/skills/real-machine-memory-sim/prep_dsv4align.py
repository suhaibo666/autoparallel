"""缩层 DeepSeek-V4 (dsv4_hybrid, unfused) 仿真 config —— 基于 v4 checkout 现成 align 配置。

关键：v4 用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid`
+ `force_unfused_dsa: true`（deepseek_v4 config 类未在 harness checkout 注册，用 v3 类 + dsv4 变体）。

在 v4 checkout 的 test_deepseekv4 目录下运行（含 dsv4_align_naive_fsdp.yaml 作 base），
PYTHONPATH 含 v4 仓库根（为 tests.utils）：
    SIM_LAYERS=4 SIM_STEPS=3 python prep_dsv4align.py
产出：同目录 dsv4_sim.yaml + train_dataset_v4/ 合成数据（seq 取 base 的 seq_length）。
"""
import os
import yaml

CUR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(CUR, "dsv4_align_naive_fsdp.yaml")
DS_DIR = os.path.join(CUR, "train_dataset_v4")
DS_FILE = os.path.join(DS_DIR, "dataset.mindrecord")

N = int(os.environ.get("SIM_LAYERS", "4"))
STEPS = int(os.environ.get("SIM_STEPS", "3"))

with open(BASE) as f:
    cfg = yaml.safe_load(f)

SEQ = int(cfg["model"].get("seq_length", 2048))

# compress_ratios 长度 = num_layers（此 config num_nextn_predict_layers=0）；混合覆盖 滑窗(0)/CSA(4)/HCA(128)
_CYCLE = [0, 4, 128]
compress_ratios = [_CYCLE[i % 3] for i in range(N)]

# 1) 合成数据集（seq 与 base 对齐）
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

# 2) 缩层 + 指向合成数据 + 少步 + 不存 ckpt
cfg["model"]["num_hidden_layers"] = N
cfg["model"]["csa_compress_ratios"] = compress_ratios
cfg["training"]["steps"] = STEPS
cfg["checkpoint"]["enable_save"] = False
cfg["checkpoint"]["load_path"] = ""          # 不加载 align ckpt
cfg["train_dataset"]["dataloader"]["dataset_files"] = [DS_DIR]
cfg["recompute"]["full_recompute_layer"] = [f"0-{N - 1}"]
# 确保 unfused 可跑（无 hyper_parallel）
cfg["model"]["force_unfused_dsa"] = True
cfg["model"]["apply_dsa_kernel_fusion"] = False

out = os.path.join(CUR, "dsv4_sim.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f, indent=2, sort_keys=False)
print("WROTE_CONFIG", out, "layers", N, "seq", SEQ, "compress_ratios", compress_ratios,
      "heads", cfg["model"]["num_attention_heads"], "v_head", cfg["model"]["v_head_dim"])
