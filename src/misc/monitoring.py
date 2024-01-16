import gc
import os
import pprint

import psutil


def memory():
    process = psutil.Process(os.getpid())
    if process.parent().name() == process.name():  # Use parent thread instead
        used = sum(p.memory_percent() for p in process.parent().children())
    else:
        used = process.memory_percent()
    s_mem = psutil.virtual_memory()
    # pprint.pprint(gc.get_stats())
    # gc.set_debug(gc.DEBUG_SAVEALL)
    # if gc.garbage:
    #     pprint.pprint(gc.garbage)
    return dict(used=used,
                system=s_mem.percent)
