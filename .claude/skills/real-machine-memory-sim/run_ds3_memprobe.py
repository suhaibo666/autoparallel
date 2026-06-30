"""跑 DeepSeek3 pynative 训练，结束后按 rank 报告真实峰值显存（真机内存仿真）。

用法（msrun 多卡，每 rank 一个进程，各自打印）：
    msrun --worker_num=2 --local_worker_num=2 --master_port=<p> --log_dir=<dir> --join=True \
        run_ds3_memprobe.py --config ds3_sim.yaml

输出每 rank：[MEMPROBE] rank=R peak_alloc_MiB=A peak_reserved_MiB=B framework_reserve_MiB=B-A
A 对标 cost_eval 评估器 report.per_stage[stage].peak_bytes；(B-A) 标定 HardwareSpec.framework_reserve。
"""
import argparse
import mindspore as ms
from mindformers.pynative.trainer import Trainer as PynativeTrainer

MIB = 2 ** 20


def _rank() -> int:
    try:
        from mindspore.communication import get_rank, GlobalComm
        if getattr(GlobalComm, "INITED", False):
            return get_rank()
    except Exception:
        pass
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    trainer = PynativeTrainer(config=args.config)
    trainer.train()

    # 峰值取整个进程的 max（含建模/优化器/训练步），即设备真实峰值
    alloc = ms.runtime.max_memory_allocated() / MIB
    reserved = ms.runtime.max_memory_reserved() / MIB
    print(f"[MEMPROBE] rank={_rank()} peak_alloc_MiB={alloc:.1f} "
          f"peak_reserved_MiB={reserved:.1f} framework_reserve_MiB={reserved - alloc:.1f}",
          flush=True)
    # rank0 dump 详细拆解（调试评估器用，可注释）
    if _rank() == 0:
        try:
            print("[MEMSUMMARY-BEGIN]\n" + ms.runtime.memory_summary() + "\n[MEMSUMMARY-END]",
                  flush=True)
        except Exception as e:
            print(f"[MEMSUMMARY] unavailable: {e}", flush=True)


if __name__ == "__main__":
    main()
