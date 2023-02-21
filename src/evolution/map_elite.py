import logging
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from random import Random
from typing import Iterable, Optional, Sequence, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdpy import tools, containers, algorithms
from qdpy.algorithms import Evolution, QDAlgorithmLike
from qdpy.containers import Container
from qdpy.phenotype import Individual as QDPyIndividual, IndividualLike, Fitness as QDPyFitness, \
    Features as QDPyFeatures
from qdpy.plots import plot_evals, plot_iterations

from abrain.core.genome import Genome, GIDManager
from .common import Individual, normalize_run_parameters


class QDIndividual(Individual, QDPyIndividual):
    def __init__(self, genome: Genome, **kwargs):
        Individual.__init__(self, genome=genome, **kwargs)
        QDPyIndividual.__init__(self)
        assert self.id() is not None
        self.name = str(self.id())

    @property
    def fitness(self):
        return QDPyFitness(list(self.fitnesses.values()),
                           [1 for _ in self.fitnesses])

    @fitness.setter
    def fitness(self, _): pass

    @property
    def features(self):
        return QDPyFeatures(list(self.descriptors.values()))

    @features.setter
    def features(self, _): pass


class Algorithm(Evolution):
    def __init__(self, container: Container, options, labels, **kwargs):
        # Manage run id, seed, data folder...
        normalize_run_parameters(options)
        name = f"ID{options.seed}"

        self.rng = Random(options.seed)
        random.seed(options.seed)
        np.random.seed(options.seed % (2**32-1))

        self.id_manager = GIDManager()

        def select(grid):
            # return self.rng.choice(grid)
            k = min(len(grid), options.tournament)
            candidates = self.rng.sample(grid.items, k)
            candidate_cells = [grid.index_grid(c.features) for c in candidates]
            curiosity = [grid.curiosity[c] for c in candidate_cells]
            if sum(curiosity) == 0:
                cell = self.rng.choice(candidate_cells)
            else:
                curiosity = [x/sum(curiosity) for x in curiosity]
                cell = self.rng.choices(candidate_cells, curiosity)[0]
            selection = candidates[candidate_cells.index(cell)]
            return selection

        def init(_):
            genome = Genome.random(self.rng, self.id_manager)
            for _ in range(options.initial_mutations):
                genome.mutate(self.rng)
            return QDIndividual(genome)

        def vary(parent):
            child = QDIndividual(parent.genome.mutated(self.rng, self.id_manager))
            self._curiosity_lut[child.id()] = self.container.index_grid(parent.features)
            return child

        sel_or_init = partial(tools.sel_or_init, init_fn=init, sel_fn=select, sel_pb=1)

        run_folder = Path(options.run_folder)
        if options.overwrite and run_folder.exists():
            shutil.rmtree(options.run_folder, ignore_errors=True)
            logging.warning(f"Purging contents of {options.run_folder}, as requested")

        run_folder.mkdir(parents=True, exist_ok=False)

        self.labels = labels
        self._curiosity_lut = {}

        Evolution.__init__(self, container=container, name=name,
                           budget=options.budget, batch_size=options.batch_size,
                           select_or_initialise=sel_or_init, vary=vary,
                           optimisation_task="maximisation",
                           **kwargs)

    def tell(self, individual: IndividualLike, *args, **kwargs) -> bool:
        grid: Grid = self.container
        added = super().tell(individual, *args, **kwargs)
        parent = self._curiosity_lut.pop(individual.id(), None)
        if parent is not None:
            grid.curiosity[parent] += {True: 1, False: -.5}[added]
        return added


class Grid(containers.Grid):
    def __init__(self, **kwargs):
        containers.Grid.__init__(self, **kwargs)
        self.curiosity = {x: 0 for x in self._index_grid_iterator()}

    def update(self, iterable: Iterable,
               ignore_exceptions: bool = True, issue_warning: bool = True) -> int:
        added = containers.Grid.update(self, iterable, ignore_exceptions, issue_warning)
        return added

    def add(self, individual: IndividualLike,
            raise_if_not_added_to_depot: bool = False) -> Optional[int]:
        r = containers.Grid.add(self, individual, raise_if_not_added_to_depot)
        return r


class Logger(algorithms.TQDMAlgorithmLogger):

    final_filename = "final.p"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, final_filename=Logger.final_filename)

    def _started_optimisation(self, algo: QDAlgorithmLike) -> None:
        """Do a mery dance so that tqdm uses stdout instead of stderr"""
        sys.stderr = sys.__stdout__
        super()._started_optimisation(algo)
        self._tqdm_pbar.file = sys.stdout

    def _vals_to_cols_title(self, content: Sequence[Any]) -> str:
        header = algorithms.AlgorithmLogger._vals_to_cols_title(self, content)
        mid_rule = "-" * len(header)
        return header + "\n" + mid_rule

    # def _tell(self, algo: QDAlgorithmLike, ind: IndividualLike) -> None:
    #     super()._tell(algo, ind)

    def summary_plots(self, **kwargs):
        summary_plots(evals=self.evals, iterations=self.iterations,
                      grid=self.algorithms[0].container,
                      labels=self.algorithms[0].labels,
                      output_dir=self.log_base_path,
                      **kwargs)


def plot_grid(data, filename, xy_range, cb_range, labels, fig_size, cmap="inferno",
              fontsize=12, nb_ticks=5):
    fig, ax = plt.subplots(figsize=fig_size)
    # cax = drawGridInAx(data, ax, cmap=cmap,
    #                    featuresBounds=featuresBounds, fitnessBounds=fitnessBounds,
    #                    aspect=aspect, xlabel=xlabel, ylabel=ylabel, nbBins=nbBins, nbTicks=nbTicks)

    g_shape = data.shape
    cax = ax.imshow(data.T, interpolation="none", cmap=plt.get_cmap(cmap),
                    vmin=cb_range[0], vmax=cb_range[1],
                    aspect="equal",
                    origin='lower', extent=(0, g_shape[0]+.5, 0, g_shape[1]+.5))
    # ax.invert_yaxis()

    # Set labels
    def ticks(i):
        pos = np.linspace(0, g_shape[i], nb_ticks)
        return pos, [
            f"{(xy_range[i][1] - xy_range[i][0]) * x / g_shape[i] + xy_range[i][0]:g}" for x in pos
        ]

    ax.set_xlabel(labels[1], fontsize=fontsize)
    ax.set_xticks(*ticks(0))
    ax.set_yticks(*ticks(1))
    ax.set_ylabel(labels[2], fontsize=fontsize)
    ax.autoscale_view()

    ax.xaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.yaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.set_xticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[1], 1), minor=True)

    # Place the colorbar with same size as the image
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size=0.5, pad=0.15)
    cbar = fig.colorbar(cax, cax=cax2, format="%g")
    cbar.ax.tick_params(labelsize=fontsize-2)
    cbar.ax.set_ylabel(labels[0], fontsize=fontsize)

    # Write
    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.exists():
        logging.info(f"Generated {filename}")
    else:
        logging.warning(f"Failed to generate {filename}")


def summary_plots(evals: pd.DataFrame, iterations: pd.DataFrame, grid: Grid,
                  output_dir: Path,
                  labels, ext="png", fig_size=(4, 4), ticks=5):
    output_path = Path(output_dir)
    def path(name): return output_path.joinpath(f"{name}.{ext}")

    plot_evals(evals["max0"], path("evals_fitnessmax0"), ylabel="Fitness")
    ylim_contsize = (0, len(grid)) if np.isinf(grid.capacity) else (0, grid.capacity)
    plot_evals(evals["cont_size"], path("evals_contsize"), ylim=ylim_contsize, ylabel="Container size")
    plot_iterations(iterations["nb_updated"], path("iterations_nbupdated"), ylabel="Number of updated bins")

    for name, cb_label, data, bounds in [
        ("performancesGrid", labels[0], grid.quality_array[..., 0], grid.fitness_domain[0]),
        ("activityGrid", "activity", grid.activity_per_bin, (0, np.max(grid.activity_per_bin)))
    ]:
        plot_path = path(name)
        plot_grid(data=data, filename=plot_path,
                  xy_range=grid.features_domain, cb_range=bounds, labels=[cb_label, *labels[1:]],
                  fig_size=fig_size, nb_ticks=5)
