"""真机 FlashAttentionScore 算子探针：确认 fwd 输出（尤其 softmax_max/softmax_sum）的
shape 与 dtype，验证评估器 `_fa_stats` 的 [2,B,N,S,8] fp32 建模是否与真实 FA 一致。

env: FA_B/FA_N/FA_S/FA_D（默认 1/8/4096/128）。单卡即可。
真机跑法（116/shb.ms.2.9，source set_env 后）:
    python fa_probe.py
"""
import math
import os

import numpy as np
import mindspore as ms
from mindspore import Tensor

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

B = int(os.environ.get("FA_B", "1"))
N = int(os.environ.get("FA_N", "8"))
S = int(os.environ.get("FA_S", "4096"))
D = int(os.environ.get("FA_D", "128"))
print(f"[FA_PROBE] B={B} N={N} S={S} D={D} dtype=bf16 layout=BNSD")


def rand(*shape):
    return Tensor(np.random.randn(*shape).astype(np.float32), ms.bfloat16)


q, k, v = rand(B, N, S, D), rand(B, N, S, D), rand(B, N, S, D)

from mindspore.ops.operations.nn_ops import FlashAttentionScore

# 因果注意力：sparse_mode=2/pre=S/next=0 常见；本探针只关心输出 shape，与 mask 无关。
fa = FlashAttentionScore(head_num=N, keep_prob=1.0, scale_value=1.0 / math.sqrt(D),
                         pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                         input_layout="BNSD", sparse_mode=0)

# 10 位置输入：query,key,value,real_shift,drop_mask,padding_mask,attn_mask,prefix,
#              actual_seq_qlen,actual_seq_kvlen —— 可选项传 None。
outs = fa(q, k, v, None, None, None, None, None, None, None)

names = ["softmax_max", "softmax_sum", "softmax_out", "attention_out"]
print("[FA_PROBE] FlashAttentionScore forward outputs:")
for nm, o in zip(names, outs):
    try:
        shp = tuple(o.shape)
        dt = str(o.dtype)
        numel = 1
        for x in shp:
            numel *= x
        print(f"[FA_OUT] {nm:14s} shape={shp} dtype={dt} numel={numel}")
    except Exception as e:
        print(f"[FA_OUT] {nm:14s} <unavailable: {e}>")

# 评估器口径对照：softmax_max + softmax_sum 各 [B,N,S,8] fp32 → 2*B*N*S*8*4 字节
model_bytes = 2 * B * N * S * 8 * 4
print(f"[FA_MODEL] _fa_stats 建模 = 2×[B,N,S,8]fp32 = {model_bytes} B = {model_bytes/2**20:.3f} MiB")
