import logging
import logging
import math
import os
import pprint
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from random import Random
from typing import Optional, List, Dict, Tuple

import abrain
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, collections
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mujoco import MjData, MjModel
from pyrr import Vector3
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, ModularRobot, Module

from ..misc.genome import RVGenome
from ..misc.config import Config
from ..simulation.control import ANNControl, SensorControlData
from ..simulation.runner import Runner, RunnerOptions


# ==============================================================================
# Robots
# ==============================================================================

def default_body() -> Body:
    BT = Config.BodyType
    b_type = Config.body_type  #BT.SNAKE

    def add_arms(m: Module):
        m.left = ActiveHinge(math.pi / 2.0)
        m.left.attachment = ActiveHinge(math.pi / 2.0)
        m.left.attachment.attachment = Brick(0.0)
        m.right = ActiveHinge(math.pi / 2.0)
        m.right.attachment = ActiveHinge(math.pi / 2.0)
        m.right.attachment.attachment = Brick(0.0)

    body = Body()
    if b_type == BT.GECKO:
        body.core.front = Brick(0.0)
        body.core.back = Brick(0.0)
        for side in ['front', 'back']:
            brick = Brick(0.0)
            setattr(body.core, side, brick)
            add_arms(brick)

    elif b_type == BT.TORSO:
        for segment in [body.core]:
            add_arms(segment)
        body.core.back = Brick(0.0)

    elif b_type == BT.SPIDER:
        body.core.front = Brick(0.0)
        body.core.back = Brick(0.0)
        for side in ['front', 'right', 'back', 'left']:
            hinge = ActiveHinge(math.pi / 2.0)
            setattr(body.core, side, hinge)
            hinge.attachment = ActiveHinge(math.pi / 2.0)
            hinge.attachment.attachment = Brick(0.0)

    elif b_type == BT.SNAKE:
        body.core.back = ActiveHinge(math.pi / 2.0)
        current = body.core.back
        for i in range(9):
            current.attachment = ActiveHinge(0)#math.pi / 2.0)
            current = current.attachment
    else:
        raise ValueError(f"Unknown body type {b_type}")

    body.finalize()
    return body


Runner.actorController_t = SensorControlData


# Runner.environmentActorController_t = ActorControl


def build_robot(brain_dna: RVGenome, with_labels: bool):
    return (ModularRobot(default_body(),
                        ANNControl.Brain(brain_dna, with_labels))
            .make_actor_and_controller())


# ==============================================================================
# Scenario
# ==============================================================================

class Scenario:
    _rewards = [     -1,     1,     0]
    #               R      G      B
    Color = Tuple[float, float, float]
    _colors: dict[str, Color] = {
        'R': [1, 0, 0],
        'G': [0, 1, 0],
        'B': [0, 0, 1],
        'Y': [1, 1, 0],
        'M': [1, 0, 1],
        'C': [0, 1, 1],
    }
    _items: Tuple[Color, Color]
    _items_pos = [.5, .5]

    def __init__(self, runner: Runner, run_id: Optional[int] = None):
        self.runner = runner
        self.id = run_id
        self.collected = None
        self._initial_position = self.subject_position()
        self._prev_position = self._initial_position
        self._steps = 0
        self._speed = 0

        if runner.options.log_path:
            self.path_log_path = (
                runner.options.save_folder.joinpath(runner.options.log_path))
            logging.info(f"Logging path to {self.path_log_path}")
            self.path_logger = open(self.path_log_path, 'w')
            self.path_logger.write(f"X Y\n")

    def finalize(self):
        if self.runner.options.log_path:
            # self.path_logger.write(f"{self.collected is not None}"
            #                        f" {self.local_fitness()[1]}")
            self.path_logger.close()
            logging.info(f"Generated {self.path_log_path}")

    @classmethod
    def aggregate(cls, folders, fitnesses, options):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        n = len(folders)
        n_rows = math.floor(math.sqrt(n))
        n_cols = n // n_rows
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 sharex="all", sharey="all",
                                 subplot_kw=dict(box_aspect=1))
        for ax, f in zip(axes.flatten(), folders):
            spec = f.stem
            df = pd.read_csv(f.joinpath(options.log_path), sep=' ')
            contact, success = fitnesses[spec]
            # df = df.iloc[:-1].astype(float)
            ax.add_collection(collections.EllipseCollection(
                widths=2*Config.item_size, heights=2*Config.item_size,
                angles=0, units='xy', facecolors=list(f.stem.lower()),
                offsets=[(cls._items_pos[0], i*cls._items_pos[1])
                         for i in [1, -1]],
                offset_transform=ax.transData
            ))

            color = "k"
            if contact:
                color = {-1: 'r', 1: 'g'}.get(success, 'b')

            ax.plot(df.X, df.Y, color=color)

            hgs = .5 * Config.ground_size
            ax.set_xlim(-hgs, hgs)
            # ax.set_xlabel("X")
            ax.set_ylim(-hgs, hgs)
            # ax.set_ylabel("Y")
            ax.set_title(f"{spec}: {success:g}")
            # ax.set_box_aspect(1)

        fitness = cls.fitness(fitnesses, Config.env_specifications)

        path = options.save_folder.joinpath(options.log_path).with_suffix('.png')
        fig.suptitle(f"Fitness: {fitness:g}")
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        print("Generated", path)

    def subject_position(self):
        return self.runner.get_actor_state(0).position

    def pre_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        pass

    def post_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        collected = set()
        for a, b in {(mj_data.geom(i), mj_data.geom(j)) for i, j in
                     zip(mj_data.contact.geom1, mj_data.contact.geom2)}:
            if "robot" in a.name and "Item" in b.name:
                collected.add(b)
            if "robot" in b.name and "Item" in a.name:
                collected.add(a)

        if len(collected) > 0:
            assert len(collected) == 1
            color_name = next(iter(collected)).name.split('#')[1]
            self.collected = (color_name, self._steps)
            self.runner.running = False

        p0 = self._prev_position
        p1 = self.subject_position()
        self._speed += (
                dt * math.sqrt((p0[0] - p1[0]) ** 2 + ((p0[1] - p1[1]) ** 2)))
        self._prev_position = p1

        if self.runner.options.log_path:
            self.path_logger.write(f"{p1.x} {p1.y}\n")

        self._steps += 1

    def process_video_frame(self, frame: np.ndarray):
        if (v := self.runner.controller.actor_controller.vision) is not None:
            ratio = .25
            w, h, _ = frame.shape
            raw_vision = v.img
            vision_ratio = raw_vision.shape[0] / raw_vision.shape[1]
            iw, ih = int(ratio * w), int(ratio * h * vision_ratio)
            scaled_vision = cv2.resize(
                cv2.cvtColor(np.flipud(raw_vision), cv2.COLOR_RGBA2BGR),
                (iw, ih),
                interpolation=cv2.INTER_NEAREST
            )
            frame[h-ih:h, w-iw:w] = scaled_vision

    # ==========================================================================

    @staticmethod
    def initial_position():
        return [0, 0, 0]
        # return [-2, 0, 0]

    # ==========================================================================

    @staticmethod
    def fitness_name():
        return "identify"

    @classmethod
    def _fitness(cls, color_name: str):
        color = cls._colors[color_name]
        value = sum(c * v for c, v in zip(color, cls._rewards))
        # print(f"[kgd-debug] {color=} {value=}")
        return value

    @classmethod
    def local_fitness_bounds(cls):
        return [(-1, 1)]

    def local_fitness(self) -> Tuple[bool, float]:
        if self.collected:
            return True, self._fitness(self.collected[0])
        else:
            x0, y0, _ = self.initial_position()
            x1, y1, _ = self.subject_position()
            x, y = self._items_pos
            def d(i0, j0, i1, j1): return math.sqrt((i1-i0)**2 + (j1-j0)**2)

            d_min = min([
                d(x1, y1, x, i*y) / d(x0, y0, x, i*y) for i in [-1, 1]
            ])
            score = .1 * (1 - d_min)
            return False, score

    @classmethod
    def fitness_bounds(cls):
        return [(-100, 100)]

    @classmethod
    @lru_cache(maxsize=1)
    def _fitness_bounds(cls, specs):
        values = [(cls._fitness(s[0]), cls._fitness(s[1])) for s in specs]
        f_min = sum(min(v0, v1) for v0, v1 in values)
        f_max = sum(max(v0, v1) for v0, v1 in values)
        assert -f_min == f_max
        return f_max

    @classmethod
    def fitness(cls, fitnesses, specs):
        f_max = cls._fitness_bounds(specs)
        score = 100 * np.sum([t[1] for t in fitnesses.values()]) / f_max
        if not np.any([t[0] for t in fitnesses.values()]):
            score -= 100
        return max(-100, score)

    # ==========================================================================

    @staticmethod
    @lru_cache(maxsize=1)
    def descriptor_names():
        return Scenario.descriptors(None, abrain.ANN())

    @staticmethod
    def descriptors(genome, brain: abrain.ANN) -> Dict[str, float]:
        stats = brain.stats()
        dct = dict()

        # dct["neurons"] = math.log(stats.hidden)/math.log(8)\
        #     if stats.hidden > 0 else 0

        def x(n: abrain.ANN.Neuron): return n.pos.tuple()[0]

        sides = [
            np.sign(x(n) * x(link.src()))
            for n in brain.neurons() for link in n.links()
        ]
        dct["ipsilateral"] = np.average(sides) if sides else 0

        inputs, outputs = len(brain.ibuffer()), len(brain.obuffer())
        if stats.hidden == 0:
            max_edges = inputs * outputs
        else:
            max_edges = (inputs * stats.hidden
                         + stats.hidden * stats.hidden
                         + stats.hidden * outputs)
        dct["connectivity"] = stats.edges / max_edges if max_edges else 0

        return dct

    @staticmethod
    @lru_cache(maxsize=1)
    def descriptor_bounds():
        dct = {"neurons": (0, Config.abrain.maxDepth),
               "connectivity": (0, 1),
               "ipsilateral": (-1, 1)}
        return [dct[k] for k in Scenario.descriptor_names()]

    # ==========================================================================

    @classmethod
    def amend(cls, xml, options: RunnerOptions):
        robots = [r for r in xml.worldbody.body]

        xml.visual.map.znear = ".001"

        # Reference to the ground
        ground = next(item for item in xml.worldbody.geom if item.name == "ground")
        g_size = Config.ground_size
        gh_size = .5 * g_size

        i_size = Config.item_size
        i_radius = .5 * i_size

        # Remove default ground
        ground.remove()

        # Add texture and material for the ground
        xml.asset.add('texture', name="grid", type="2d", builtin="checker",
                      width="512", height="512",
                      rgb1=".1 .2 .3", rgb2=".2 .3 .4")
        xml.asset.add('material', name="grid", texture="grid",
                      texrepeat="1 1", texuniform="true", reflectance="0")
        xml.worldbody.add('geom', name="floor",
                          size=[gh_size, gh_size, .05],
                          type="plane", material="grid", condim=3)

        xml.asset.add('texture', name="sky", type="skybox", builtin="flat",
                      rgb1="0 0 0", rgb2="0 0 0",
                      width=512, height=512,
                      mark="random", markrgb="1 1 1")
        #
        # gh_width = .025
        # for i, x, y in [(0, 0, 1), (1, 1, 0), (2, 0, -1), (3, -1, 0)]:
        #     b_height = g_size / 100
        #     xml.worldbody.add('geom', name=f"border#{i}",
        #                       pos=[x * (gh_size + gh_width),
        #                            y * (gh_size + gh_width), b_height],
        #                       rgba=[1, 1, 1, 1],
        #                       euler=[0, 0, i * math.pi / 2],
        #                       type="box", size=[gh_size, gh_width, b_height])

        assert len(options.current_specs) == 2
        for spec, side in zip(options.current_specs, [1, -1]):
            name = f"Item#{spec}"

            body = xml.worldbody.add('body', name=name,
                                     pos=[cls._items_pos[0],
                                          cls._items_pos[1] * side,
                                          0])
            body.add('geom', name=name,
                     rgba=[*cls._colors[spec], 1],
                     type="sphere", size=f"{i_radius} 0 0")

        if options is not None and options.view is not None and options.view.mark_start:
            for robot in robots:
                xml.worldbody.add('site',
                                  name=robot.full_identifier[:-1] + "_start",
                                  pos=robot.pos * [1, 1, 0], rgba=[0, 0, 1, 1],
                                  type="ellipsoid", size=[0.05, 0.05, 0.0001])

        for r in robots:
            for g in r.find_all('geom'):
                if math.isclose(g.size[0], g.size[1]):
                    g.rgba = ".3 0 .3 1"
                else:
                    g.rgba = "0 .3 0 1"
                    # g.material = creeper_material

        for hinge in filter(lambda j: j.tag == 'joint', xml.find_all('joint')):
            xml.sensor.add('jointpos', name=f"{hinge.full_identifier}_sensor".replace('/', '_'),
                           joint=hinge.full_identifier)
