"""单卡显存采集自检（在 116/shb.ms.2.9 容器内、source set_env.sh 后运行）。

验证 ms.runtime 的峰值显存 API 可用，确认采集机制在真机 Ascend 上工作。
实测（2026-06-30, MindSpore 2.10）：分配 128MiB fp16 + x*2 → max_allocated≈256 MiB, max_reserved≈258 MiB。

运行：
    ssh 192.168.9.116 'docker exec shb.ms.2.9 bash -lc \
      "source /usr/local/Ascend/ascend-toolkit/set_env.sh; python -"' < probe_device_mem.py
或在容器内：source set_env.sh && python probe_device_mem.py
"""
import os
import numpy as np
import mindspore as ms

DEV = int(os.environ.get("PROBE_DEVICE", "0"))   # 选空闲卡：PROBE_DEVICE=<id>
MIB = 2 ** 20

ms.set_device("Ascend", DEV)
ms.runtime.reset_peak_memory_stats()

# 在设备上实际占用约 128 MiB，再做一次 elementwise 触发第二块分配
x = ms.Tensor(np.ones((1024, 1024, 64), np.float16))   # 1024*1024*64*2 = 128 MiB
y = x * 2
ms.runtime.synchronize()

alloc = ms.runtime.max_memory_allocated() / MIB
reserved = ms.runtime.max_memory_reserved() / MIB
print(f"[PROBE] device={DEV} max_allocated_MiB={alloc:.1f} max_reserved_MiB={reserved:.1f}")
print(f"[PROBE] framework_reserve_estimate_MiB={reserved - alloc:.1f}  (= reserved - allocated)")
# 完整文本拆解（可解析各类占用）：
# print(ms.runtime.memory_summary())
