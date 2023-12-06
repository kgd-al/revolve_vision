#!/usr/bin/env python3
import json
import logging
import os
import pprint
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.default_experiment.scenario import build_robot
from src.evolution.common import Individual

os.environ["QT_QPA_PLATFORM"] = "offscreen"

start = time.perf_counter()

assert len(sys.argv) > 1

N = 1
stats_keys = ["axons", "depth", "hidden", "edges", "iterations"]
cppn_keys = ["nodes", "links"]
time_keys = ["build_t", "eval_t"]

df = pd.DataFrame(columns=cppn_keys + stats_keys + time_keys + ["file"])

ann_files = dict()
o_folder = Path("tmp/ann_time_tests")
o_folder.mkdir(parents=True, exist_ok=True)

for file in sys.argv[1:]:
    # print(file)

    ind = Individual.from_file(file)
    stats = dict()
    for k in stats_keys:
        stats[k] = []
    times = dict(build=[], eval=[])

    for i in range(N):
        _, controller = build_robot(ind.genome, True)
        ann = controller.brain

        rng = random.Random(0)
        inputs, outputs = ann.buffers()
        for _ in range(1000):
            inputs[:] = np.random.uniform(-1, 1, len(inputs))
            ann(inputs, outputs)

        ann_stats = ann.stats().dict()
        for k in stats_keys:
            stats[k].append(ann_stats[k])
        times["build"].append(ann_stats["time"]["build"])
        times["eval"].append(ann_stats["time"]["eval"])

    for k in stats_keys:
        assert len(set(stats[k])) == 1
        stats[k] = stats[k][0]
    for k in times:
        times[k] = np.average(times[k])

    # for k in stats_keys:
    #     if ind.stats[k] != stats[k]:
    #         logging.warning(f"[{file}:{k}] {ind.stats[k]} != {stats[k]}")

    df.loc[len(df)] = [
        len(ind.genome.brain.nodes), len(ind.genome.brain.links)
    ] + list(stats.values()) + list(times.values()) + [file]

    ind.stats = stats
    ind.stats["time"] = times

    hidden = stats["hidden"]
    o_file_id = ann_files.get(hidden, 0)
    ann_files[hidden] = o_file_id+1

    j = ind.to_json()
    del j['fitnesses']
    del j['descriptors']
    del j['eval_time']
    with open(o_folder.joinpath(f"ann_{hidden}_{o_file_id}.json"), "w") as f:
        json.dump(j, f)

print(df)

for lhs_k in cppn_keys + stats_keys:
    for rhs_k in time_keys:
        plot = df.plot.scatter(x=lhs_k, y=rhs_k)
        fig = plot.figure
        fig.suptitle(f"{lhs_k} / {rhs_k}")
        fig.tight_layout()
        fig.savefig(o_folder.joinpath(f"{lhs_k}_vs_{rhs_k}.png"),
                    bbox_inches='tight')
        plt.close(fig)

duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
print(f"Generated ANN stats for {len(sys.argv)-1} files in {duration}")
