import argparse
import json
import pprint
from inspect import isclass
from pathlib import Path
from typing import Annotated, get_type_hints

import abrain


class Config:
    simulation_time: Annotated[float, "Duration of the simulation"] = 10
    control_frequency: Annotated[float, "How frequently to call the controller"] = 10

    ground_size: Annotated[float, "Total size of the arena"] = 5
    item_range_max: Annotated[float, "Maximal distance from starting point to an object"] = 1
    item_range_min: Annotated[float, "Minimal distance from starting point to an object"] = .5
    item_size: Annotated[float, "Size of an object"] = .1

    brain_types = [abrain.ANN.__name__, "CPG"]
    brain_type: Annotated[float, f"The type of brain to use {brain_types}"] = abrain.ANN.__name__
    # brain_type = abrain.ANN.__name__
    abrain = abrain.Config

    @classmethod
    def write_json(cls, file: Path):
        with open(file, 'w') as f:
            json.dump(Config._get_items(Config), f, indent=2)
            f.write('\n')

    @classmethod
    def read_json(cls, file: Path):
        with open(file, 'r') as f:
            Config._set_items(Config, json.load(f))

    @classmethod
    def print(cls):
        pprint.pprint(cls._items())

    @classmethod
    def argparse_setup(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Config", f"Alter configuration values (see {__file__})")
        hints = get_type_hints(cls, include_extras=True)
        for k, v in cls._get_items(cls).items():
            hint = hints.get(k)
            if hint is not None:
                group.add_argument('--config-' + k, dest="config_" + k, metavar='V',
                                   type=type(v), help='.'.join(hint.__metadata__))

    @classmethod
    def argparse_process(cls, args):
        for k, v in filter(lambda i: i[0].startswith('config_'), args.__dict__.items()):
            if v is not None:
                setattr(cls, k[7:], v)

    @classmethod
    def _items(cls):
        return Config._get_items(Config)

    @staticmethod
    def _set_items(cls, dct):
        for k, v in dct.items():
            attr = getattr(cls, k, None)
            if not isclass(attr):
                setattr(cls, k, v)

    @staticmethod
    def _get_items(cls):
        return {k: v_ for k in dir(cls) if (v_ := Config._value(cls, k))}

    @staticmethod
    def _value(cls, key):
        attr = getattr(cls, key)
        if key.startswith("_"):
            attr = None
        elif isclass(attr):
            if attr == abrain.Config:
                attr = abrain.Config.to_json()
            else:
                attr = Config._get_items(attr)
        elif callable(attr):
            attr = None

        return attr
