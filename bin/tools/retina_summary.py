#!/usr/bin/env python3
import ast
import json
import math
import pprint
import pickle
import sys
from pathlib import Path
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib import patches
from matplotlib.axes import Axes
from scipy.stats import mannwhitneyu


def main():
    base_folder = Path("remote/identify_v1")
    if len(sys.argv) > 1:
        print(sys.argv)
        base_folder = Path(sys.argv[1])

    data = []
    hist_data, time_data = {}, {}
    groups = set()
    budgets, iterations = set(), set()
    folders = list(base_folder.glob("*100K"))
    assert len(folders) > 0, f"No run folder found under {base_folder}"
    for folder in folders:
        file = folder.joinpath("best.json")
        if not file.exists():
            continue

        with open(file, 'r') as fd:
            d = json.load(fd)

        rid, group, _ = folder.stem.split('-')
        groups.add(group)
        d.update({"group": group})
        data.append(d)

        def _append(lhs, rhs):
            for k_, v_ in rhs.items():
                if k_ not in lhs:
                    lhs[k_] = []
                lhs[k_].append(v_)

        _append(hist_data, dict(
            rid=rid, group=group,
            gid=d["id"],
            fitness=d["fitnesses"]["identify"],
            # vision=reduce(lambda x, y: x * y, d["genome"]["vision"].values()),
            depth=d["stats"]["depth"],
            n_hidden=d["stats"]["hidden"],
            # n_total=d["stats"]["neurons"],
            n_links=d["stats"]["edges"],
            cppn_lid=d["genome"]["brain"]["LID"],
            cppn_nid=d["genome"]["brain"]["NID"],
            cppn_ln=len(d["genome"]["brain"]["links"]),
            cppn_nn=len(d["genome"]["brain"]["nodes"]),
        ))

        file = folder.joinpath("iteration-final.p")
        with open(file, 'rb') as fd:
            d = pickle.load(fd)

        budgets.add(d["budget"])
        iterations.add(len(d["iterations"]))

        def _append(lhs, rhs):
            for k_, v_ in rhs.items():
                if k_ not in lhs:
                    lhs[k_] = {}
                if group not in lhs[k_]:
                    lhs[k_][group] = np.c_[v_]
                else:
                    lhs[k_][group] = np.c_[lhs[k_][group], v_]

        def _fmt(k, f): return [f(v_) for v_ in d["iterations"][k]]
        def _norm(l_): return 100 * float(l_[0]) / float(l_[1])
        _append(time_data, dict(
            cont_size=_fmt("cont_size", lambda o: _norm(str(o).split("/"))),
            max=_fmt("max", lambda o: ast.literal_eval(o)[0]),
            avg=_fmt("avg", lambda o: ast.literal_eval(o)[0]),
            qd_score=_fmt("qd_score", float),
        ))

    groups = sorted(groups)

    assert len(budgets) == 1
    budget = next(iter(budgets))
    assert len(iterations) == 1
    iterations = next(iter(iterations))
    index = np.linspace(budget/iterations, budget, iterations)

    # pprint.pprint(data)
    df = pd.DataFrame(data=hist_data)
    df.to_csv(base_folder.joinpath("hist.csv"), sep=' ', index=False)
    # pprint.pprint(df)

    for c in df.columns[2:]:
        v_min, v_max = np.quantile(df[c], [0, 1])
        hist = df.hist(column=c, by='group', sharex=True, sharey=True,
                       layout=(len(groups), 1),
                       bins=10, range=(v_min, v_max))

        gdf = df.groupby("group")[c]
        for i, g_lhs in enumerate(groups):
            for g_rhs in groups[i+1:]:
                _, p = mannwhitneyu(gdf.get_group(g_lhs), gdf.get_group(g_rhs))
                if p <= .05:
                    print(f"mannwhitneyu({c:10s}, {g_lhs}, {g_rhs}): p={p}")
        hist[-1].set_xlabel(c)
        file = base_folder.joinpath(f"hist_{c}.png")
        plt.savefig(file)
        write_log(file)

    colors = ["C1", "C2", "C3"]
    def color(i_): return colors[i_ % len(colors)]

    plt.rcParams["figure.figsize"] = (15, 7)

    labels = {
        "cont_size": "Container size (%)",
        "max": "Max fitness",
        "avg": "Average fitness",
        "qd_score": "QD Score",
    }

    # quantiles = [0, .05, .125, .25, .325, .5, .625, .75, .875, .95, 1]
    quantiles = [0, .05, .25, .5, .75, .95, 1]
    alpha_inc = 1 / (len(quantiles) // 2 + 1)

    # pprint.pprint(time_data)
    variables = list(time_data.keys())
    # print(f"{variables=}")
    # print(f"{groups=}")
    for v in variables:
        n = len(groups)
        n_rows = math.floor(math.sqrt(n))
        n_cols = n // n_rows
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols+1,
                                 sharex='all', sharey='all')
        fig.supxlabel("Evaluations")
        fig.supylabel(labels[v])
        for i, (ax, g) in enumerate(zip(axes[:, :-1].flatten(), groups)):
            ax.set_title(g)
            ax.set_axisbelow(True)
            ax.yaxis.grid(linestyle='dashed')
            d = time_data[v][g]
            # pprint.pprint(d)
            qtls = np.quantile(d, quantiles, axis=1,
                               method="closest_observation")

            handles = []
            for j in range(len(quantiles) // 2):
                ax.fill_between(index, qtls[j], qtls[-j-1],
                                alpha=alpha_inc, color=color(i))
                handles.append(
                    patches.Patch(color=color(i),
                                  alpha=alpha_inc*(j+1),
                                  label=f"{100*quantiles[j]:g}"
                                        f"-{100*quantiles[-j-1]:g}%"))
            handles.append(
                ax.plot(index, qtls[3], color=color(i), label="50%")[0])
            ax.legend(handles=handles)

        for ax, group in zip(axes[:, -1].flatten(), zip(*(iter(groups),) * 3)):
            subgroup = np.concatenate([time_data[v][g] for g in group], axis=1)
            avg, std = np.average(subgroup, axis=1), np.std(subgroup, axis=1)
            ax.yaxis.grid(linestyle='dashed')

            for i, g in enumerate(group):
                ax.plot(index, np.average(time_data[v][g], axis=1),
                        color=color(i), alpha=.25,
                        label=g)
            ax: Axes
            ax.plot(index, avg, color="C0", label="Avg.")
            ax.legend()

        file = base_folder.joinpath(f"time_{v}.png")
        fig.tight_layout()
        fig.savefig(file)
        write_log(file)


def write_log(file):
    if file.exists():
        print("Generated", file)
    else:
        print(f"{Fore.RED}Error generating {file}{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
