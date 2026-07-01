"""跑 DSv3 pynative 训练 + MindSpore Profiler(profile_memory=True)，dump 真机内存 timeline。

对标 cost_eval 的 `timeline_probe.py`（仿真逐事件曲线）：本脚本采**真机**逐时刻 allocated/reserved
曲线，产出 <prof_out>/rank_<r>/ASCEND_PROFILER_OUTPUT/memory_record.csv（内存随时间）
+ operator_memory.csv（逐算子内存）。并保留 [MEMPROBE] 峰值行做交叉核对。

用法（msrun 多卡）：
    msrun --worker_num=2 --local_worker_num=2 --master_port=<p> --log_dir=<dir> --join=True \
        run_ds3_memtimeline.py --config ds3_sim.yaml --prof_out <dir>/prof_mem
"""
import argparse
import os
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
    p.add_argument("--prof_out", default="./prof_mem")
    args = p.parse_args()

    # 先构造 trainer（内部 set context/device、init hccl），再起 Profiler 覆盖 train() 全程
    trainer = PynativeTrainer(config=args.config)
    prof = None
    try:
        prof = ms.Profiler(output_path=args.prof_out, profile_memory=True)
        print(f"[PROF] rank={_rank()} profiler started -> {args.prof_out}", flush=True)
    except Exception as e:
        print(f"[PROF] rank={_rank()} Profiler init failed: {e}", flush=True)

    trainer.train()

    if prof is not None:
        try:
            prof.analyse()
            print(f"[PROF] rank={_rank()} analyse done", flush=True)
        except Exception as e:
            print(f"[PROF] rank={_rank()} analyse failed: {e}", flush=True)

    alloc = ms.runtime.max_memory_allocated() / MIB
    reserved = ms.runtime.max_memory_reserved() / MIB
    print(f"[MEMPROBE] rank={_rank()} peak_alloc_MiB={alloc:.1f} "
          f"peak_reserved_MiB={reserved:.1f} framework_reserve_MiB={reserved - alloc:.1f}",
          flush=True)
    print(f"[PROF_OUT] rank={_rank()} {os.path.abspath(args.prof_out)}", flush=True)


if __name__ == "__main__":
    main()
