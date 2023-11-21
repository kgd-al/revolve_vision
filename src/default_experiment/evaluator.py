import logging
import math
import pprint
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import abrain
from abrain.core.ann import plotly_render
from ..misc.config import Config
from .scenario import build_robot, Scenario
from ..evolution.common import EvaluationResult
from ..misc.genome import RVGenome
from ..simulation.control import OpenGLVision
from ..simulation.runner import Runner, RunnerOptions, CallbackType, \
    ANNDataLogging

logger = logging.getLogger(__name__)


@dataclass
class EvalOptions:
    runner: RunnerOptions = field(default_factory=RunnerOptions)
    specs: Optional[str | tuple[str]] = None

    ann_save_path: Optional[Path] = None


class Evaluator:
    options = EvalOptions()

    @classmethod
    def set_runner_options(cls, options: RunnerOptions):
        cls.options.runner = options

    @dataclass
    class Result(EvaluationResult):
        # fitnesses: Individual.DataCollection = field(default_factory=dict)
        # descriptors: Individual.DataCollection = field(default_factory=dict)
        # stats: Individual.DataCollection = field(default_factory=dict)
        reject: bool = False

    @staticmethod
    def _clip(values: Dict[str, float], bounds: List[Tuple[float]], name: str):
        for i, b in zip(values.items(), bounds):
            k, v, lower, upper = *i, *b
            if not lower <= v <= upper:
                values[k] = max(lower, min(v, upper))
                logger.warning(f"Out-of-bounds value {name},{k}: {v} not in [{lower},{upper}]")
        return values

    @classmethod
    def _evaluate(cls, genome: RVGenome, options: EvalOptions, rerun: bool):
        # pprint.pprint(options)

        r = cls.Result()
        viewer = None

        if options.specs:
            if isinstance(options.specs, str):
                specs = tuple(options.specs)
            else:
                specs = options.specs
        else:
            specs = Config.env_specifications

        with_labels = (options.ann_save_path is not None
                       or options.runner.ann_data_logging != ANNDataLogging.NONE)
        robot_body, robot_brain = build_robot(genome, with_labels)
        brain: abrain.ANN = robot_brain.brain

        if (p := options.ann_save_path) is not None:
            p: Path = options.runner.save_folder.joinpath(p)
            pr = plotly_render(brain, robot_brain.labels)
            pr.write_html(p)
            logging.info(f"Generated {p}")

            # if False:
            #     p_ = p.with_suffix(".mp4")
            #     frames = 100
            #
            #     xd = dict(title=None, visible=False, showticklabels=False)
            #     pr.update_layout(title=dict(automargin=False),
            #                      scene=dict(xaxis=xd, yaxis=xd, zaxis=xd),
            #                      margin=dict(autoexpand=False, b=0, l=0, r=0, t=0),
            #                      width=500, height=500)
            #     for i in range(frames):
            #         cr = 2
            #         ct = 2 * math.pi * i / frames
            #         eye = dict(x=cr * math.cos(ct), y=cr * math.sin(ct), z=0)
            #         print(f"{100*i/frames:03.2f}% {eye}")
            #         pr.update_layout(scene_camera=dict(eye=eye))
            #         pr.write_image(f"test_image_{i:03d}.png")
            #     logging.info(f"Generated images")

        r.reject = brain.empty()

        r.stats = brain.stats().dict()
        r.stats.update({
            'brain': float(not brain.empty()),
            'perceptron': float(brain.perceptron()),
            'neurons': len(brain.neurons()),
        })

        r.stats['items'] = {}
        fitnesses = dict()

        save_folder = options.runner.save_folder
        save_folders = []

        if not r.reject:
            for spec in specs:
                options.runner.current_specs = spec

                if save_folder:
                    options.runner.save_folder = save_folder.joinpath(spec)
                    options.runner.save_folder.mkdir(parents=True,
                                                     exist_ok=True)
                    save_folders.append(options.runner.save_folder)

                runner = Runner(robot_body, robot_brain, options.runner,
                                Scenario.amend,
                                position=Scenario.initial_position())
                scenario = Scenario(runner, genome.id())

                runner.callbacks[CallbackType.PRE_CONTROL_STEP] = (
                    scenario.pre_control_step)
                runner.callbacks[CallbackType.POST_CONTROL_STEP] = (
                    scenario.post_control_step)

                if options.runner.record is not None:
                    runner.callbacks[CallbackType.VIDEO_FRAME_CAPTURED] = (
                        scenario.process_video_frame)

                brain_controller = runner.controller.actor_controller
                assert brain is brain_controller.brain
                brain.reset()

                if genome.with_vision():
                    brain_controller.vision = \
                        OpenGLVision(runner.model, genome.vision,
                                     runner.headless)

                if rerun:
                    viewer = runner.viewer

                runner.run()

                scenario.finalize()

                local_fitness = scenario.local_fitness()
                fitnesses[spec] = local_fitness

                r.stats['items'][spec] = (local_fitness, scenario.collected)

        options.runner.save_folder = save_folder
        if not r.reject and save_folder:
            Scenario.aggregate(save_folders, fitnesses, options.runner)

        r.fitnesses = {Scenario.fitness_name():
                       Scenario.fitness(fitnesses, Config.env_specifications)}
        r.descriptors = cls._clip(Scenario.descriptors(genome, brain),
                                  Scenario.descriptor_bounds(),
                                  "features")

        if rerun:
            return r, viewer
        else:
            return r

    @classmethod
    def evaluate_evo(cls, genome: RVGenome) -> Result:
        return cls._evaluate(genome, cls.options, False)

    @classmethod
    def evaluate_rerun(cls, genome: RVGenome, options: EvalOptions):
        return cls._evaluate(genome, options, True)
