import os

import psutil

import nvidia_smi


def memory():
    process = psutil.Process(os.getpid())
    if process.parent().name() == process.name():  # Use parent thread instead
        used = sum(p.memory_info().rss for p in process.parent().children())
    else:
        used = process.memory_info().rss
    s_mem = psutil.virtual_memory()

    # pprint.pprint(gc.get_stats())
    # gc.set_debug(gc.DEBUG_SAVEALL)
    # if gc.garbage:
    #     pprint.pprint(gc.garbage)

    # print(guppy.hpy().heap())

    return {
        "python-used": used,
        "system-used": s_mem.total - s_mem.available,
        "system-total": s_mem.total,
        **gpu_usage()
    }


def gpu_usage():
    nvidia_smi.nvmlInit()
    devices = nvidia_smi.nvmlDeviceGetCount()
    usage = {"gpu-used": [], "gpu-total": []}
    for i in range(devices):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        usage['gpu-used'].append(mem.used)
        usage['gpu-total'].append(mem.total)
    nvidia_smi.nvmlShutdown()
    usage = {
        k: sum(v) for k, v in usage.items()
    }
    return usage
