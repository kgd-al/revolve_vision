import logging
import math
import os
import pprint
from functools import lru_cache
from typing import Optional, Dict, Tuple

import abrain
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, collections
from mujoco import MjData, MjModel
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, ModularRobot, Module

from ..misc.config import Config
from ..misc.genome import RVGenome
from ..simulation.control import ANNControl, SensorControlData
from ..simulation.runner import Runner, RunnerOptions

logger = logging.getLogger(__name__)


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

        self.path_logger = None
        if runner.options.path_log_file:
            self.path_log_path = (
                runner.options.save_folder.joinpath(runner.options.path_log_file))
            logger.debug(f"Logging path to {self.path_log_path}")
            self.path_logger = open(self.path_log_path, 'w')
            self.path_logger.write(f"X Y\n")

    def finalize(self):
        if self.path_logger:
            # self.path_logger.write(f"{self.collected is not None}"
            #                        f" {self.local_fitness()[1]}")
            self.path_logger.close()
            logger.debug(f"Generated {self.path_log_path}")

        if Config.debug_retina_brain:
            self._debug_retina(end=True)

    @classmethod
    def aggregate(cls, folders, fitnesses, options):
        path_file = options.path_log_file
        if path_file:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

            n = len(folders)
            n_rows = math.floor(math.sqrt(n))
            n_cols = n // n_rows
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                     sharex="all", sharey="all",
                                     subplot_kw=dict(box_aspect=1))
            for ax, f in zip(axes.flatten(), folders):
                spec = f.stem
                df = pd.read_csv(f.joinpath(path_file), sep=' ')
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
                ax.set_title(f"{spec}: {round(success, 4):g}")
                # ax.set_box_aspect(1)

            fitness = cls.fitness(fitnesses, Config.env_specifications)

            path = options.save_folder.joinpath(options.log_path).with_suffix('.png')
            fig.suptitle(f"Fitness: {fitness:g}")
            fig.tight_layout()
            fig.savefig(path, bbox_inches='tight')
            logger.info(f"Generated {path}")

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

        if self.runner.options.path_log_file:
            self.path_logger.write(f"{p1.x} {p1.y}\n")

        if Config.debug_retina_brain:
            self._debug_retina()

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

            # (v0) Strict version -> rewards getting close to the correct object
            if Config.fitness == "v0":
                i = np.argmax([self._fitness(s) for s
                               in self.runner.options.current_specs])
                y *= i * -1
                d = d(x1, y1, x, y) / d(x0, y0, x, y)

            # (v1) Nice version -> rewards getting close to any object
            elif Config.fitness == "v1":
                d = min([
                    d(x1, y1, x, i*y) / d(x0, y0, x, i*y) for i in [-1, 1]
                ])

            else:
                d = 0
                logger.error("No fitness config value!")

            score = .1 * (1 - d)
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

    def _debug_retina(self, end=False):
        controller = self.runner.controller.actor_controller
        iw, ih = controller.vision.width, controller.vision.height
        if not end:
            brain = controller.brain

            if self._steps == 0:

                self._img_logger = pd.DataFrame(
                    columns=[f"{c}{i}{j}" for j in range(ih) for i in range(iw) for c in "RGB"])

                self._inputs_logger = pd.DataFrame(
                    columns=[f"I{i}" for i in range(len(brain.ibuffer()))]
                )

            img = controller.vision.img
            # pprint.pprint(img)
            # pprint.pprint([[[f"{c}{i}{j}" for c in "RGB"] for i in range(iw)] for j in range(ih)])
            # print(list(self._img_logger.columns))
            # print(img.flatten())
            self._img_logger.loc[self._steps] = [x / 255 for x in img.flatten()]
            cv2.imwrite(str(self.runner.options.save_folder
                            .joinpath(f'vision_{self._steps:010d}.png')),
                        cv2.cvtColor(np.flipud(img),
                                     cv2.COLOR_RGBA2BGR))

            inputs = brain.ibuffer()
            self._inputs_logger.loc[self._steps] = inputs.tuple()

        else:
            self._img_logger.to_csv(
                self.runner.options.save_folder.joinpath('debug.img.dat'))
            self._inputs_logger.to_csv(
                self.runner.options.save_folder.joinpath('debug.inputs.dat'))
            self._neurons_logger = controller.monitor.neurons_data

            # print(self._img_logger.columns)
            # print(self._inputs_logger.columns)
            # print(self._neurons_logger.columns)

            img_neurons_mapper = {
                label: s
                for s in self._neurons_logger.columns
                if (label := "".join(c for c in s.split(":")[1]
                                     if c not in "[,]"))
                in self._img_logger.columns
            }
            img_inputs_mapper = {
                label: iid
                for label, iid in zip(self._img_logger.columns,
                                      self._inputs_logger.columns[-3*iw*ih:])
            }
            # pprint.pprint(img_neurons_mapper)

            fig_size = 3
            fig = plt.figure(figsize=(3*iw*fig_size, ih*fig_size),
                             layout="constrained")
            sub_figs = fig.subfigures(1, 3, wspace=.02)
            axes = [sub_fig.subplots(ih, iw, sharex='all', sharey='all',
                                     gridspec_kw={'wspace': 0, 'hspace': .25})
                    for sub_fig in sub_figs.ravel()]

            assert (len(self._img_logger) == len(self._inputs_logger)
                    == len(self._neurons_logger))

            for c, sub_axes in zip("RGB", axes):
                columns = [
                    col for col in self._img_logger.columns if col[0] == c
                ]
                columns = sorted(columns, key=lambda c: (-int(c[2]), int(c[1])))
                for c_img, ax in zip(columns, sub_axes.flatten()):
                    c_input = img_inputs_mapper[c_img]
                    c_neuron = img_neurons_mapper[c_img]
                    # print(c_img, c_input, c_neuron, ax)
                    for data, name, color in [
                            (self._img_logger[c_img], "Image", 'm'),
                            (self._inputs_logger[c_input], "Inputs", 'y'),
                            (self._neurons_logger[c_neuron], "Neurons", 'c')]:
                        ax.plot(data, label=name, color=color, alpha=.33)
                    ax.set_ylim([0, 1])
                    ax.grid()
                    ax.set_title(f"{c_img} | {c_input} | {c_neuron}",
                                 fontsize=10)

            handles, labels = ax.get_legend_handles_labels()
            # print(f"{handles=}")
            # print(f"{labels=}")
            fig.legend(handles, labels, loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.05))

            # fig.tight_layout()
            fig.savefig(self.runner.options.save_folder.joinpath("debug.png"),
                        bbox_inches='tight')

            print(self.runner.options.save_folder)
        # if self._steps == 0:
        #     os.environ["QT_QPA_PLATFORM"] = "offscreen"
        #
        #     brain = controller.brain
        #     neurons = {n.pos: n for n in brain.neurons()}
        #
        #     self.__retina_neurons = {k: [] for k in "RGB"}
        #     for p, lbl in controller.labels.items():
        #         if lbl[0] in "RGB":
        #             self.__retina_neurons[lbl[0]].append(
        #                 (p.tuple(), neurons[p], lbl))
        #     # pprint.pprint(self.__retina_neurons)
        #
        # aspect = controller.vision.height / controller.vision.width
        # fig, axes = plt.subplots(2, 3,
        #                          sharex='all', sharey='all',
        #                          figsize=(10, 10*aspect))
        #
        # img = controller.vision.img
        # cv2.imwrite(str(self.runner.options.save_folder
        #                 .joinpath(f'vision_{self._steps:010d}.png')),
        #             cv2.cvtColor(np.flipud(img),
        #                          cv2.COLOR_RGBA2BGR))
        #
        # img_extents = (-1.5, 1.5, -1.5, 1.5)
        # cmaps = {"R": lambda v: [v, 0, 0],
        #          "G": lambda v: [0, v, 0],
        #          "B": lambda v: [0, 0, v]}
        #
        # for i, (k, layer) in enumerate(self.__retina_neurons.items()):
        #     # pprint.pprint(list(zip(*layer)))
        #     ps, ns, lbls = zip(*layer)
        #     xs, ys, zs = zip(*ps)
        #
        #     cmap = cmaps[k]
        #     vs = [cmap(n.value) for n in ns]
        #     print(k, vs)
        #
        #     ax = axes[1, i]
        #     # ax.set_box_aspect(aspect)
        #     ax.scatter(xs, zs, c=vs, edgecolors='white')
        #     for x, z, lbl in zip(xs, zs, lbls):
        #         # print(lbl, x, z)
        #         dz = 1 if z < 1 else -1
        #         ax.annotate(lbl, (x, z), (0, 1.5 * dz),
        #                     bbox=dict(facecolor='white', alpha=1),
        #                     textcoords='offset fontsize',
        #                     ha='center', va='center')
        #     ax.imshow(img[:, :, i], extent=img_extents,
        #               cmap='gray', vmin=0, vmax=255)
        #
        # axes[0, 1].imshow(img, extent=img_extents)
        #
        # axes[0, 0].axis('off')
        # axes[0, 2].axis('off')
        #
        # fig.tight_layout()
        # fig.savefig(self.runner.options.save_folder
        #             .joinpath(f"vision_{self._steps:010d}.png"),
        #             bbox_inches='tight')
        # plt.close()
        # # exit(42)
        #
        # # pprint.pprint(img)
        # #
        # # neurons = {k: [
        # #     brain.neurons()[(x, y, z)]
        # # ] for k, layer in self.__layers.items()}

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
                      rgb1=".1 .1 .1", rgb2=".2 .2 .2")
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
