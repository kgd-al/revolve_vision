#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import field
from datetime import timedelta
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abrain._cpp.phenotype
from src.default_experiment.scenario import build_robot
from src.evolution.common import Individual
from src.misc.config import Config

import numpy.polynomial.polynomial as npoly

class Options:
    def __init__(self):
        self.evaluate: bool = True
        self.plot: bool = True
        self.iterations: int = 10
        self.debug: bool = False
        self.files: list[str] = field(default_factory=list)

        self.o_folder = Path("tmp/ann_time_tests")

        self.tests = [True, False]

        self.stats_keys = ["hidden", "depth", "axons", "edges", "density", "iterations"]
        self.cppn_keys = ["nodes", "links"]
        self.time_keys = [f"{t}_{n}_t" for t in ["build", "eval"] for n in ["ann", "cppn"]]

    def dataframe(self): return self.o_folder.joinpath("stats.csv")

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument('-N', '--iterations', type=int,
                            help="Specify number of iterations used to"
                                 " smoothen out stochasticity")
        parser.add_argument('--no-evaluate', action='store_false',
                            dest='evaluate', help="Use existing data")
        parser.add_argument('--no-plot', action='store_false',
                            dest='plot', help="Do not generate plots")
        parser.add_argument('--debug', action='store_true',
                            help="Write debug data to logs (huge slow down)")
        parser.add_argument('files', metavar="FILE",
                            nargs="+", help="Individual file to process")

        Config.argparse_setup(parser)


def main():
    args = Options()
    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    args.o_folder.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        df = evaluate(args)
    else:
        df = pd.read_csv(args.dataframe())

    print(df.sort_values(by=["file", "impl"]).to_string(max_rows=1000))

    if args.plot:
        plot(args, df)


def plot(args, df):
    if len(args.tests) == 2:
        groups = df.groupby(by="impl")
        for lhs_k in args.cppn_keys + args.stats_keys:
            for rhs_k in args.time_keys:
                fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
                coefficients = {}
                for key, ax in zip(groups.groups.keys(), axes.flatten()):
                    g = groups.get_group(key)
                    x, y = g[lhs_k], g[rhs_k]
                    x_min, x_max = np.quantile(x, [0, 1])

                    g.plot.scatter(x=lhs_k, y=rhs_k, ax=ax)

                    r = npoly.Polynomial.fit(x, y, deg=1, full=True)
                    b, a = r[0]
                    ax.plot(*r[0].linspace(100, ax.get_xlim()),
                            color='C1', label='ax+b')

                    coefficients[key] = [a, b]

                    ax.set_title(f"{key}: {a:.3g}x + {b:.3g}")
                    ax.set_yscale('log')

                def summarize():
                    return ", ".join(
                        f"{k}={100 * coefficients['py'][i] / coefficients['cpp'][i]:.0f}%"
                        for i, k in enumerate(["a", "b"])
                    )

                fig.suptitle(f"{lhs_k} / {rhs_k}: ({summarize()})")
                fig.tight_layout()
                fig.savefig(args.o_folder.joinpath(f"{lhs_k}_vs_{rhs_k}.png"),
                            bbox_inches='tight')
                plt.close(fig)


def evaluate(args):
    start = time.perf_counter()

    df = pd.DataFrame(
        columns=args.cppn_keys + args.stats_keys + args.time_keys + ["impl", "file"])

    ann_files = dict()

    tests = [True, False]
    for pure_python in tests:
        abrain.use_pure_python(pure_python)
        print(
            f"Monkey-patching abrain for pure python: {pure_python}\n"
            + "\n".join(f"{c}" for c in [abrain.ANN, abrain.Point]))
        flag = "py" if pure_python else "cpp"
        logger = logging.getLogger(flag)
        file_handler = logging.FileHandler(args.o_folder.joinpath(
            flag + ".log"), mode="w")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        if args.debug:
            logger.debug(flag)

        if pure_python:
            def pretty_point(p_: abrain.Point):
                return "{" + ",".join(f"{v:+.3g}" for v in p_.__iter__()) + "}"

            def pretty_outputs(outs):
                return [o for o in outs]
        else:
            def pretty_point(p_: abrain.Point):
                return "{" + ",".join(f"{v:+.3g}" for v in p_.tuple()) + "}"

            def pretty_outputs(outs):
                return [outs[i_] for i_ in range(len(outs))]

        for i, file in enumerate(args.files):
            print(i, file)

            ind = Individual.from_file(file)
            stats = dict()
            for k in args.stats_keys:
                stats[k] = []
            times = {k: [] for k in args.time_keys}

            for i in range(args.iterations):
                start_time = abrain._python.ann._time()
                cppn = abrain.CPPN(ind.genome.brain)
                times["build_cppn_t"].append(abrain._python.ann._time_diff(start_time))

                rng = random.Random(0)

                times["eval_cppn_t"].append(0)
                cppn_outputs = cppn.outputs()
                for _ in range(1000):
                    start_time = abrain._python.ann._time()

                    def p():
                        return abrain.Point(*[rng.uniform(-1, 1) for _ in range(3)])

                    p0, p1 = p(), p()
                    cppn(p0, p1, cppn_outputs)

                    if args.debug:
                        logger.debug(f"p0={pretty_point(p0)}"
                                     f" p1={pretty_point(p1)}"
                                     f" outputs={pretty_outputs(cppn_outputs)}")

                    times["eval_cppn_t"][-1] += abrain._python.ann._time_diff(start_time)

                _, controller = build_robot(ind.genome, True)
                ann = controller.brain

                inputs, outputs = ann.buffers()
                for _ in range(1000):
                    inputs[:] = np.random.uniform(-1, 1, len(inputs))
                    ann(inputs, outputs)

                ann_stats = ann.stats().dict()
                for k in args.stats_keys:
                    stats[k].append(ann_stats[k])
                times["build_ann_t"].append(ann_stats["time"]["build"])
                times["eval_ann_t"].append(ann_stats["time"]["eval"])

            for k in args.stats_keys:
                assert len(set(stats[k])) == 1
                stats[k] = stats[k][0]
            for k in times:
                times[k] = np.average(times[k])

            # for k in stats_keys:
            #     if ind.stats[k] != stats[k]:
            #         logging.warning(f"[{file}:{k}] {ind.stats[k]} != {stats[k]}")

            df.loc[len(df)] = [
                                  len(ind.genome.brain.nodes), len(ind.genome.brain.links)
                              ] + list(stats.values()) + list(times.values()) + [
                                  flag, file
                              ]

            ind.stats = stats
            ind.stats["time"] = times

            hidden = stats["hidden"]
            o_file_id = ann_files.get(hidden, 0)
            ann_files[hidden] = o_file_id + 1

            j = ind.to_json()
            del j['fitnesses']
            del j['descriptors']
            del j['eval_time']
            with open(args.o_folder.joinpath(f"ann_{hidden}_{o_file_id}.json"), "w") as f:
                json.dump(j, f)

            # break

    df.to_csv(args.dataframe())

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Generated ANN stats for {len(sys.argv) - 1} files in {duration}")

    return df


if __name__ == '__main__':
    main()
