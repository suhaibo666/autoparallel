"""MemoryProbeCallback —— 在 mindformers PyNative 训练中采集每 rank 真实峰值显存。

挂点：mindformers/pynative/callback/callback.py:22 TrainerCallback
      钩子 on_step_begin / on_step_end(self, args, state, **kwargs)（参考 LossCallback / MaxLogitsMonitor）。

用法（在 116/shb.ms.2.9 容器内，source set_env.sh 后）：把本类加入 trainer 的 callbacks。
    from mem_probe_callback import MemoryProbeCallback
    trainer = Trainer(config, callbacks=[MemoryProbeCallback(warmup=3)])  # 或按该版本注册方式
    trainer.train()

它在第 `warmup` 步 begin 处 reset 峰值统计，在第 `warmup+1` 步 end 处读取并按 rank 打印：
    [MEMPROBE] rank=R peak_alloc_MiB=A peak_reserved_MiB=B
A 对标 cost_eval 评估器的 per_stage[stage].peak_bytes；(B - A) 用于标定 HardwareSpec.framework_reserve。
"""
import mindspore as ms

try:
    from mindformers.pynative.callback import TrainerCallback
except Exception:                      # 包结构兜底
    from mindformers.pynative.callback.callback import TrainerCallback

MIB = 2 ** 20


def _global_step(state) -> int:
    """容错读取步计数：不同版本 state 属性名可能不同。"""
    for attr in ("global_step", "cur_step", "step", "global_step_num", "cur_step_num"):
        v = getattr(state, attr, None)
        if isinstance(v, int):
            return v
    return -1


def _rank() -> int:
    try:
        from mindspore.communication import get_rank, GlobalComm
        if getattr(GlobalComm, "INITED", False):
            return get_rank()
    except Exception:
        pass
    return 0


class MemoryProbeCallback(TrainerCallback):
    """warmup 步后清零峰值，下一步结束读取 max_memory_allocated/reserved 并按 rank 打印。"""

    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self._done = False

    def on_step_begin(self, args, state, **kwargs):
        if _global_step(state) == self.warmup:
            ms.runtime.reset_peak_memory_stats()

    def on_step_end(self, args, state, **kwargs):
        if self._done:
            return
        step = _global_step(state)
        if step >= self.warmup + 1:
            alloc = ms.runtime.max_memory_allocated() / MIB
            reserved = ms.runtime.max_memory_reserved() / MIB
            print(f"[MEMPROBE] rank={_rank()} step={step} "
                  f"peak_alloc_MiB={alloc:.1f} peak_reserved_MiB={reserved:.1f} "
                  f"framework_reserve_MiB={reserved - alloc:.1f}", flush=True)
            self._done = True
