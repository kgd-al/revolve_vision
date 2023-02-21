import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import abrain
from abrain import Genome
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from .scenario import build_robot, Scenario
from ..evolution.common import Individual
from ..misc.config import Config
from ..simulation.runner import Runner, RunnerOptions, CallbackType

logger = logging.getLogger(__name__)


class Evaluator:
    options: RunnerOptions = None

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
    def _evaluate(cls, genome: Genome, options: RunnerOptions, rerun: bool):
        r = cls.Result()
        viewer = None

        robot = build_robot(genome)
        is_abrain = (Config.brain_type == abrain.ANN.__name__)
        if not is_abrain:
            # Default CPG does not use external (sensor) data -> use default controller
            Runner.environmentActorController_t = EnvironmentActorController

        runner = Runner(robot, options, Scenario.amend)
        scenario = Scenario(runner)

        runner.callbacks[CallbackType.POST_CONTROL_STEP] = scenario.post_control_step

        r.reject = False
        if is_abrain:
            brain: abrain.ANN = runner.controller.actor_controller.brain
            r.reject |= (brain.empty())

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

        if rerun:
            return r, viewer
        else:
            return r

    @classmethod
    def evaluate_evo(cls, genome: Genome) -> Result:
        return cls._evaluate(genome, cls.options, False)

    @classmethod
    def evaluate_rerun(cls, genome: Genome, options: RunnerOptions):
        return cls._evaluate(genome, options, True)
