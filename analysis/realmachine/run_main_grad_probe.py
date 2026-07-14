"""Measure accumulated gradients at the PyNative optimizer boundary on Ascend."""
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


def _dtype_bytes(dtype) -> int:
    name = str(dtype).lower()
    if "64" in name:
        return 8
    if "32" in name:
        return 4
    if "16" in name:
        return 2
    if "8" in name or "bool" in name:
        return 1
    raise TypeError(f"unsupported gradient dtype: {dtype}")


def _snapshot(models, phase: str) -> None:
    seen_params: set[int] = set()
    param_numel = 0
    grad_numel = 0
    grad_bytes = 0
    grad_buffers = 0
    main_grad_buffers = 0
    param_grad_buffers = 0
    samples: list[str] = []

    for model in models:
        for param in model.get_parameters():
            identity = id(param)
            if identity in seen_params:
                continue
            seen_params.add(identity)
            param_numel += math.prod(param.shape)
            main_grad = getattr(param, "main_grad", None)
            param_grad = getattr(param, "grad", None)
            grad = main_grad if main_grad is not None else param_grad
            if grad is None:
                continue
            source = "main_grad" if main_grad is not None else "param.grad"
            main_grad_buffers += int(main_grad is not None)
            param_grad_buffers += int(main_grad is None and param_grad is not None)
            numel = math.prod(grad.shape)
            nbytes = numel * _dtype_bytes(grad.dtype)
            grad_buffers += 1
            grad_numel += numel
            grad_bytes += nbytes
            if len(samples) < 5:
                samples.append(f"{param.name}:{source}:{tuple(grad.shape)}:{grad.dtype}")

    current_alloc = ms.runtime.memory_allocated()
    print(
        f"[MAIN_GRAD] rank={_rank()} phase={phase} buffers={grad_buffers} "
        f"main_grad_buffers={main_grad_buffers} param_grad_buffers={param_grad_buffers} "
        f"grad_numel={grad_numel} grad_bytes={grad_bytes} "
        f"grad_MiB={grad_bytes / MIB:.1f} param_numel={param_numel} "
        f"current_alloc_MiB={current_alloc / MIB:.1f} samples={'|'.join(samples)}",
        flush=True,
    )


class MainGradProbeTrainer(PynativeTrainer):
    """Observe gradients before the real optimizer consumes and clears them."""

    def _optimizer_update(self):
        ms.runtime.synchronize()
        _snapshot(self.model, "before_optimizer")
        return super()._optimizer_update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    trainer = MainGradProbeTrainer(config=args.config)
    trainer.train()
    ms.runtime.synchronize()
    _snapshot(trainer.model, "after_zero_grad")

    peak_alloc = ms.runtime.max_memory_allocated()
    peak_reserved = ms.runtime.max_memory_reserved()
    print(
        f"[MEMPROBE] rank={_rank()} peak_alloc_MiB={peak_alloc / MIB:.1f} "
        f"peak_reserved_MiB={peak_reserved / MIB:.1f} "
        f"framework_reserve_MiB={(peak_reserved - peak_alloc) / MIB:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
