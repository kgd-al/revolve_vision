#!/usr/bin/env python3

import argparse
import logging
import pickle
import pprint
import sys
from pathlib import Path
from typing import Tuple, Optional

from qdpy.plots import plot_evals

from src.evolution.map_elite import Logger, summary_plots


class Options:
    def __init__(self):
        self.src_folder: Path
        self.dst_folder: Optional[Path] = None

        self.iteration: str = Logger.final_filename
        self.format: str = "png"
        self.fig_size: Tuple[float] = (4., 4.)

    @staticmethod
    def populate(p: argparse.ArgumentParser):
        group = p.add_argument_group("Storage",
                                     "Defines data/outputs location")
        group.add_argument('src_folder', metavar='DIR', type=Path,
                           help="Path to data folder")
        group.add_argument('--out', dest="dst_folder", metavar='DIR',
                           type=Path, help="Where to output the summaries"
                                           " (defaults to the source folder)")
        group.add_argument('--file', dest="file", metavar='FILE',
                           type=str, help="Name (not path) of the file to load")

        group = p.add_argument_group("Formats",
                                     "Defines the plots aspects")
        group.add_argument('--format', dest="format", metavar='F', choices=["png", "pdf"],
                           type=str, help="File format for the plots")
        group.add_argument('--fig-size', dest="fig_size", metavar='S',
                           type=Tuple[float], help="Size of the generated plots")


def process(options, data_file):
    data = pickle.load(data_file)
    if options.dst_folder is None:
        options.dst_folder = options.src_folder

    print(type(data), pprint.pformat(data))

    def path(name): return options.dst_folder.joinpath(f"{name}.{options.format}")

    plot_evals(data["evals"]["max0"], path("evals_fitnessmax0"), ylabel="Fitness",
               figsize=options.fig_size)

    summary_plots(evals=data["evals"], iterations=data["iterations"],
                  grid=data["container"],
                  output_dir=options.dst_folder, labels=data["labels"],
                  fig_size=args.fig_size, ext=options.format)


if __name__ == '__main__':
    args = Options()
    parser = argparse.ArgumentParser(description="Summary/plotter for QDPy map-elite")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    df_path = args.src_folder.joinpath(args.iteration)
    logging.info(f"Loading from {df_path}")
    with open(df_path, "rb") as df:
        process(args, df)
