import json
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from random import Random
from typing import Optional, List, Dict

import cv2
import mujoco
import numpy as np
from mujoco import MjData, MjModel

import abrain
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, ModularRobot
from ..misc.config import Config
from ..simulation.control import ANNControl, SensorControlData, CPGControl
from ..simulation.runner import Runner, RunnerOptions


# ==============================================================================
# Robots
# ==============================================================================

def default_body() -> Body:
    body = Body()
    body.core.front = Brick(0.0)
    body.core.back = Brick(0.0)
    for segment in [body.core.front, body.core.back]:
        segment.left = ActiveHinge(math.pi / 2.0)
        segment.left.attachment = ActiveHinge(math.pi / 2.0)
        segment.left.attachment.attachment = Brick(0.0)
        segment.right = ActiveHinge(math.pi / 2.0)
        segment.right.attachment = ActiveHinge(math.pi / 2.0)
        segment.right.attachment.attachment = Brick(0.0)
    body.finalize()
    return body


Runner.actorController_t = SensorControlData


# Runner.environmentActorController_t = ActorControl


def build_robot(brain_dna, with_labels):
    if Config.brain_type == abrain.ANN.__name__:
        brain = ANNControl.Brain(brain_dna, with_labels)
    else:
        brain = CPGControl.Brain(brain_dna)
    robot = ModularRobot(default_body(), brain)
    return robot


# ==============================================================================
# Items
# ==============================================================================


class CollectibleType(Enum):
    Apple = 0
    Pepper = 1

    @staticmethod
    def random(rng: Random):
        return rng.choice([t for t in CollectibleType])

    def _v(self, separation):
        return .5 * (1 + (2 * self.value - 1) * separation)

    def radius(self, separation: float):
        v = 1-.5*self._v(separation)
        return [1, v, v]

    def color(self, separation: float):
        v = self._v(separation)
        return [v, 1-v, 0, 1]


@dataclass
class CollectibleObject:
    x: float = 0
    y: float = 0
    type: CollectibleType = CollectibleType.Apple


# ==============================================================================
# Scenario
# ==============================================================================

class Scenario:
    _rewards = {
        str(CollectibleType.Apple): 1,
        str(CollectibleType.Pepper): -1,
    }

    _items: Optional[List[CollectibleObject]] = None

    def __init__(self, runner: Runner, run_id: Optional[int] = None):
        self.runner = runner
        self.id = run_id
        self.collected = {str(e): 0 for e in CollectibleType}
        self._initial_position = self.subject_position()
        self._prev_position = self._initial_position
        self._steps = 0
        self._speed = 0

    @staticmethod
    def initial_position():
        return [0, 0, 0]

    def subject_position(self):
        return self.runner.get_actor_state(0).position

    def pre_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        pass

    def post_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        collected = set()
        for a, b in {(mj_data.geom(i), mj_data.geom(j)) for i, j in
                     zip(mj_data.contact.geom1, mj_data.contact.geom2)}:
            if "robot" in a.name and CollectibleType.__name__ in b.name:
                collected.add(b)
            if "robot" in b.name and CollectibleType.__name__ in a.name:
                collected.add(a)

        if len(collected) > 0:
            for geom in collected:
                tokens = geom.name.split('#')
                t_id = tokens[0]
                b_id = int(tokens[1])
                mj_data.mocap_pos[b_id][2] = -.25
                self.collected[t_id] += self._rewards[t_id]

        p0 = self._prev_position
        p1 = self.subject_position()
        self._speed += dt * math.sqrt((p0[0] - p1[0]) ** 2 + ((p0[1] - p1[1]) ** 2))
        self._prev_position = p1

        self._steps += 1

    def process_video_frame(self, frame: np.ndarray):
        ratio = .25
        w, h, _ = frame.shape
        raw_vision = self._vision.img
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
    def fitness_name():
        return "speed"

    def fitness(self) -> Dict[str, float]:
        score = 0
        if self._steps > 0:
            score += 100 * self._speed / self._steps
        # score += sum([t for t in self.collected.values()])
        return {self.fitness_name(): score}

    @classmethod
    def fitness_bounds(cls):
        return [(0, 2)]

    # ==========================================================================

    @staticmethod
    def descriptor_names():
        return ["x", "y"]

    def descriptors(self) -> Dict[str, float]:
        # random = Random()
        # return [random.uniform(-2, 2) for _ in range(2)]
        return {"x": self.subject_position()[0],
                "y": self.subject_position()[1]}

    @staticmethod
    def descriptor_bounds():
        w = Config.ground_size / 2
        return [(-w, w), (-w, w)]

    # ==========================================================================
    #
    # @staticmethod
    # def descriptor_names(): return ["height", "angle"]
    #
    # def descriptors(self) -> List[float]:
    #     return [v / self._steps if self._steps > 0 else 0
    #             for v in [max(0, self._altitude), self._gait]]
    #
    # @staticmethod
    # def descriptor_bounds(): return [(0.0, .5), (-90.0, 90.0)]
    #
    # # ==========================================================================

    @staticmethod
    def sunflower(n: int, r_range) -> np.ndarray:
        # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
        k_theta = np.pi * (3 - np.sqrt(5))
        angles = np.linspace(k_theta, k_theta * n, n)

        r_min, r_max = r_range
        radii = np.sqrt(np.linspace(0, 1, n)) * (r_max - r_min) + r_min

        # Return Cartesian coordinates from polar ones.
        return (radii * np.stack((np.cos(angles), np.sin(angles)))).T

    @staticmethod
    def generate_initial_items(count=20, r_range=(.5, 1)):
        items = []
        # count=512
        # r_range=(.5,2)

        if count > 0:
            coordinates = Scenario.sunflower(count, r_range)
            for ix, iy in coordinates:
                items.append(
                    CollectibleObject(ix, iy, list(CollectibleType)[len(items) % 2]))

        Scenario._items = items

    @staticmethod
    def mutate_items(rng: Random):
        raise NotImplementedError

    @staticmethod
    def serialize(path: Path):
        with open(path, 'w') as f:
            json.dump([(i.x, i.y, i.type.value) for i in Scenario._items], f)

    @staticmethod
    def deserialize(path: Path):
        with open(path, 'r') as f:
            Scenario._items = [
                CollectibleObject(i[0], i[1], CollectibleType(i[2])) for i in json.load(f)
            ]

    # ==========================================================================

    @staticmethod
    def amend(xml, options: RunnerOptions):
        item_distinction = 1    # float(os.environ.get("separation", 1))

        robots = [r for r in xml.worldbody.body]

        xml.visual.map.znear = ".001"

        # Reference to the ground
        ground = next(item for item in xml.worldbody.geom
                      if item.name == "ground")
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

        # Funny but not currently relevant
        ts, rnd = 8, item_distinction
        xml.asset.add('texture', name=f"txt{CollectibleType.Apple.name}",
                      type="cube", builtin="flat",
                      rgb1="0 1 0", rgb2="0 1 0", width=ts, height=ts,
                      mark="random", markrgb="1 0 0", random=rnd)
        xml.asset.add('material', name=f"mat{CollectibleType.Apple.name}",
                      texture=f"txt{CollectibleType.Apple.name}",
                      texrepeat="1 1", texuniform="true", reflectance="0")
        xml.asset.add('texture', name=f"txt{CollectibleType.Pepper.name}",
                      type="cube", builtin="flat",
                      rgb1="1 0 0", rgb2="1 0 0", width=ts, height=ts,
                      mark="random", markrgb="0 1 0", random=rnd)
        xml.asset.add('material',
                      name=f"mat{CollectibleType.Pepper.name}",
                      texture=f"txt{CollectibleType.Pepper.name}",
                      texrepeat="1 1", texuniform="true", reflectance="0")

        gh_width = .025
        for i, x, y in [(0, 0, 1), (1, 1, 0), (2, 0, -1), (3, -1, 0)]:
            b_height = g_size / 100
            xml.worldbody.add('geom', name=f"border#{i}",
                              pos=[x * (gh_size + gh_width),
                                   y * (gh_size + gh_width), b_height],
                              rgba=[1, 1, 1, 1],
                              euler=[0, 0, i * math.pi / 2],
                              type="box", size=[gh_size, gh_width, b_height])

        if Scenario._items is not None:
            for i, item in enumerate(Scenario._items):
                name = f"{item.type}#{i}"
                # radii = [i_radius * 1 for r in item.type.radius(item_distinction)]

                body = xml.worldbody.add('body', name=name,
                                         pos=[item.x, item.y, 0],
                                         mocap=True)
                body.add('geom', name=name,
                         # rgba=item.type.color(item_distinction),
                         type="sphere", size=f"{i_radius} 0 0",
                         material=f"mat{item.type.name}")

                # body = xml.worldbody.add('body', name=name,
                #                          pos=[item.x, item.y, radii[-1]],
                #                          mocap=True)
                # body.add('geom', name=name,
                #          rgba=item.type.color(item_distinction),
                #          type="ellipsoid", size=radii)
                #          # material="object")

        if options is not None and options.view is not None and options.view.mark_start:
            for robot in robots:
                xml.worldbody.add('site',
                                  name=robot.full_identifier[:-1] + "_start",
                                  pos=robot.pos * [1, 1, 0], rgba=[0, 0, 1, 1],
                                  type="ellipsoid", size=[0.05, 0.05, 0.0001])

        for hinge in filter(lambda j: j.tag == 'joint', xml.find_all('joint')):
            xml.sensor.add('jointpos', name=f"{hinge.full_identifier}_sensor".replace('/', '_'),
                           joint=hinge.full_identifier)
