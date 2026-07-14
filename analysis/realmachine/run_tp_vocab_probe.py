"""TP vocab 栈探针（P0-04 真机验证，2026-07-14）：TP>1 下 embedding / output_layer 权重的
每卡 local shape 是否按 vocab 维 ÷tp（RowwiseParallel/ColwiseParallel Shard(0)，
parallelize.py:751-767），并记录峰值 allocated/reserved。只读观测，不改训练数学。"""
from __future__ import annotations

import argparse
import math

import mindspore as ms
from mindformers.pynative.trainer import Trainer as PynativeTrainer

MIB = 2**20


def _rank() -> int:
    try:
        from mindspore.communication import GlobalComm, get_rank
        if getattr(GlobalComm, "INITED", False):
            return get_rank()
    except Exception:
        pass
    return 0


def _dump_vocab_stack(models, tag: str) -> None:
    for model in models:
        for param in model.get_parameters():
            n = param.name
            if ("embedding" in n or "output_layer" in n or "lm_head" in n) and "norm" not in n:
                print(f"[TPVOCAB] rank={_rank()} {tag} {n} local_shape={tuple(param.shape)} "
                      f"local_numel={math.prod(param.shape)} dtype={param.dtype}", flush=True)


class TpVocabProbeTrainer(PynativeTrainer):
    def _optimizer_update(self):
        ms.runtime.synchronize()
        return super()._optimizer_update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    trainer = TpVocabProbeTrainer(config=args.config)
    _dump_vocab_stack(trainer.model, "after_init")
    trainer.train()
    ms.runtime.synchronize()
    peak_alloc = ms.runtime.max_memory_allocated()
    peak_reserved = ms.runtime.max_memory_reserved()
    print(f"[MEMPROBE] rank={_rank()} peak_alloc_MiB={peak_alloc / MIB:.1f} "
          f"peak_reserved_MiB={peak_reserved / MIB:.1f}", flush=True)


if __name__ == "__main__":
    main()
