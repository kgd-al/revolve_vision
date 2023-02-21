import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import Random
from typing import Optional, List, Dict

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


def build_robot(brain_dna):
    if Config.brain_type == abrain.ANN.__name__:
        brain = ANNControl.Brain(brain_dna)
    else:
        brain = CPGControl.Brain(brain_dna)
    robot = ModularRobot(default_body(), brain)
    return robot

# ==============================================================================
# Items
# ==============================================================================


class CollectibleType(Enum):
    Apple = 'sphere'
    Pepper = 'ellipsoid'

    @staticmethod
    def random(rng: Random):
        return rng.choice([t for t in CollectibleType])


def radius(c_type: 'CollectibleType'):
    return {CollectibleType.Apple: 1, CollectibleType.Pepper: .5}[c_type]


def color(c_type: 'CollectibleType'):
    return {CollectibleType.Apple: [0, 1, 0, 1],
            CollectibleType.Pepper: [1, 0, 0, 1]}[c_type]


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

    def __init__(self, runner: Runner):
        self.runner = runner
        self.collected = {str(e): 0 for e in CollectibleType}
        self._initial_position = self.subject_position()
        self._prev_position = self._initial_position
        self._steps = 0
        self._speed = 0

    def subject_position(self):
        return self.runner.get_actor_state(0).position

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
                mj_data.mocap_pos[b_id][2] = 1
                self.collected[t_id] += self._rewards[t_id]

        p0 = self._prev_position
        p1 = self.subject_position()
        self._speed += dt * math.sqrt((p0[0] - p1[0]) ** 2 + ((p0[1] - p1[1]) ** 2))
        self._prev_position = p1

        self._steps += 1

    # ==========================================================================

    @staticmethod
    def fitness_name(): return "speed"

    def fitness(self) -> Dict[str, float]:
        score = 0
        if self._steps > 0:
            score += 100*self._speed / self._steps
        # score += sum([t for t in self.collected.values()])
        return {self.fitness_name(): score}

    @classmethod
    def fitness_bounds(cls): return [(0, 2)]

    # ==========================================================================

    @staticmethod
    def descriptor_names(): return ["x", "y"]

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
    def sunflower(n: int, alpha: float) -> np.ndarray:
        # Number of points respectively on the boundary and inside the cirlce.
        n_exterior = np.round(alpha * np.sqrt(n)).astype(int)
        n_interior = n - n_exterior

        # Ensure there are still some points in the inside...
        if n_interior < 1:
            raise RuntimeError(f"Parameter 'alpha' is too large ({alpha}), all "
                               f"points would end-up on the boundary.")
        # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
        k_theta = np.pi * (3 - np.sqrt(5))
        angles = np.linspace(k_theta, k_theta * n, n)

        # Generate the radii.
        r_interior = np.sqrt(np.linspace(Config.item_range_min, Config.item_range_max, n_interior))
        r_exterior = np.full((n_exterior,), Config.item_range_max)
        r = np.concatenate((r_interior, r_exterior))

        # Return Cartesian coordinates from polar ones.
        return (r * np.stack((np.cos(angles), np.sin(angles)))).T

    @staticmethod
    def generate_initial_items(rng: Random, count=10):

        items = []
        for ix, iy in Scenario.sunflower(count, 1):
            items.append(
                CollectibleObject(ix, iy, CollectibleType.random(rng)))

        # items.append(   ## To test collisions
        #     CollectableObject(0, 0, CollectableType.Apple)
        # )

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

    @staticmethod
    def amend(xml, options: RunnerOptions):
        robots = [r for r in xml.worldbody.body]

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

        gh_width = .025
        for i, x, y in [(0, 0, 1), (1, 1, 0), (2, 0, -1), (3, -1, 0)]:
            b_height = g_size / 5
            xml.worldbody.add('geom', name=f"border#{i}",
                              pos=[x * (gh_size+gh_width),
                                   y * (gh_size+gh_width), b_height],
                              rgba=[1, 1, 1, 1],
                              euler=[0, 0, i*math.pi/2],
                              type="box", size=[gh_size, gh_width, b_height])

        if Scenario._items is not None:
            for i, item in enumerate(Scenario._items):
                name = f"{item.type}#{i}"
                body = xml.worldbody.add('body', name=name,
                                         pos=[item.x, item.y, i_radius * radius(item.type)],
                                         mocap=True)
                body.add('geom', name=name,
                         rgba=color(item.type),
                         type=item.type.value,
                         size=[i_radius, .5*i_radius, .5*i_radius])

        if options is not None and options.view is not None and options.view.mark_start:
            for robot in robots:
                xml.worldbody.add('site',
                                  name=robot.full_identifier[:-1]+"_start",
                                  pos=robot.pos * [1, 1, 0], rgba=[0, 0, 1, 1],
                                  type="ellipsoid", size=[0.05, 0.05, 0.0001])

        for hinge in filter(lambda j: j.tag == 'joint', xml.find_all('joint')):
            xml.sensor.add('jointpos', name=f"{hinge.full_identifier}_sensor".replace('/', '_'),
                           joint=hinge.full_identifier)
