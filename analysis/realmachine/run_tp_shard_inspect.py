"""TP vocab 栈**分片**探针（P0-04 决定性证据，2026-07-14）：只做 Trainer 初始化（跳过会在
fork DSA+TP kernel 崩的训练步），检查 embedding/output_layer 权重是否 DTensor 且其**本地分片**
（to_local / placements）沿 vocab 维 ÷tp。param.shape 报逻辑全局 shape，故必须看 local 存储。"""
from __future__ import annotations

import argparse
import math

from mindformers.pynative.trainer import Trainer as PynativeTrainer


def _rank() -> int:
    try:
        from mindspore.communication import GlobalComm, get_rank
        if getattr(GlobalComm, "INITED", False):
            return get_rank()
    except Exception:
        pass
    return 0


def _local_view(param):
    """尽力取 DTensor 本地分片的 shape 与 placements（不同 MindSpore 版本 API 名不同）。"""
    info = {"global_shape": tuple(param.shape)}
    for attr in ("placements", "_placements", "sharding_spec", "_sharding_spec"):
        v = getattr(param, attr, None)
        if v is not None:
            info["placements"] = str(v)
            break
    for meth in ("to_local", "_to_local", "local_tensor", "get_local_tensor"):
        fn = getattr(param, meth, None)
        if callable(fn):
            try:
                loc = fn()
                info["local_shape"] = tuple(loc.shape)
                info["local_numel"] = math.prod(loc.shape)
            except Exception as e:  # noqa: BLE001
                info["local_err"] = f"{meth}:{type(e).__name__}"
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    trainer = PynativeTrainer(config=args.config)   # 只初始化，不 train
    for model in trainer.model:
        for param in model.get_parameters():
            n = param.name
            if ("embedding" in n or "output_layer" in n or "lm_head" in n) and "norm" not in n:
                print(f"[TPSHARD] rank={_rank()} {n} {_local_view(param)}", flush=True)


if __name__ == "__main__":
    main()
