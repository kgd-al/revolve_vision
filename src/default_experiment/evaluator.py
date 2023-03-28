import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import abrain
from abrain import Genome
from abrain.core.ann import plotly_render
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from ..simulation.control import OpenGLVision
from .scenario import build_robot, Scenario
from ..evolution.common import Individual
from ..misc.config import Config
from ..simulation.runner import Runner, RunnerOptions, CallbackType

logger = logging.getLogger(__name__)


@dataclass
class EvalOptions:
    runner: RunnerOptions = RunnerOptions()

    ann_save_path: Optional[Path] = None


class Evaluator:
    options: Optional[EvalOptions] = None

    @classmethod
    def set_runner_options(cls, options: RunnerOptions):
        cls.options = options

    @dataclass
    class Result:
        fitnesses: Individual.DataCollection = field(default_factory=dict)
        descriptors: Individual.DataCollection = field(default_factory=dict)
        stats: Individual.DataCollection = field(default_factory=dict)
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
    def _evaluate(cls, genome: Genome, options: EvalOptions, rerun: bool):
        r = cls.Result()
        viewer = None

        with_labels = (options.ann_save_path is not None)

        robot = build_robot(genome, with_labels)
        is_abrain = (Config.brain_type == abrain.ANN.__name__)
        if not is_abrain:
            # Default CPG does not use external (sensor) data -> use default controller
            Runner.environmentActorController_t = EnvironmentActorController

        runner = Runner(robot, options.runner, Scenario.amend,
                        position=Scenario.initial_position())
        scenario = Scenario(runner, genome.id())

        runner.callbacks[CallbackType.PRE_CONTROL_STEP] = scenario.pre_control_step
        runner.callbacks[CallbackType.POST_CONTROL_STEP] = scenario.post_control_step

        if options.runner.record is not None:
            runner.callbacks[CallbackType.VIDEO_FRAME_CAPTURED] = scenario.process_video_frame

        r.reject = False
        assert is_abrain
        brain_controller = runner.controller.actor_controller
        if is_abrain:
            brain: abrain.ANN = brain_controller.brain
            r.reject |= (brain.empty())

        if hasattr(genome, 'vision'):
            brain_controller.vision = \
                OpenGLVision(runner.model, genome.vision, runner.headless)

        if (p := options.ann_save_path) is not None:
            plotly_render(brain, runner.controller.actor_controller.labels).write_html(p)

        if rerun:
            viewer = runner.viewer

        if not r.reject:
            runner.run()

        r.fitnesses = cls._clip(scenario.fitness(), scenario.fitness_bounds(),
                                "fitness")
        r.descriptors = cls._clip(scenario.descriptors(), scenario.descriptor_bounds(),
                                  "features")

        r.stats = {}
        if is_abrain:
            r.stats.update({
                'brain': float(not brain.empty()),
                'perceptron': float(brain.perceptron()),
                'hidden': brain.stats().hidden,
                'neurons': len(brain.neurons()),
                'edges': brain.stats().edges,
                'depth': brain.stats().depth,
            })

        if False:
            r.stats.update({
                'apple': scenario.collected[str(CollectibleType.Apple)],
                'pepr': -scenario.collected[str(CollectibleType.Pepper)],
            })

        # print(f"Evaluated\n {pprint.pformat(genome.to_json())}\n {pprint.pformat(r)}\n")

        if rerun:
            return r, viewer
        else:
            return r

    @classmethod
    def evaluate_evo(cls, genome: Genome) -> Result:
        return cls._evaluate(genome, cls.options, False)

    @classmethod
    def evaluate_rerun(cls, genome: Genome, options: EvalOptions):
        return cls._evaluate(genome, options, True)
