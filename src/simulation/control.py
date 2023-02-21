from typing import List, Tuple

import numpy as np
from revolve2.serialization import StaticData, Serializable

import abrain
from abrain import Point, CPPN
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Brain, Body, ActiveHinge
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbour
from ..simulation.runner import DefaultActorControl


class SensorControlData(DefaultActorControl):
    def __init__(self, mj_model, mj_data):
        DefaultActorControl.__init__(self, mj_model, mj_data)
        self.sensors = mj_data.sensordata


class ANNControl:
    class Controller(ActorController):
        def __init__(self, genome: abrain.Genome, inputs: List[Point], outputs: List[Point]):
            self.brain = abrain.ANN.build(inputs, outputs, genome)
            self.i_buffer, self.o_buffer = self.brain.buffers()

        def get_dof_targets(self) -> List[float]:
            return [self.o_buffer[i] for i in range(len(self.o_buffer))]

        def step(self, dt: float, data: 'ControlData') -> None:
            self.i_buffer[:] = [pos for pos in data.sensors]
            self.brain.__call__(self.i_buffer, self.o_buffer)

        @classmethod
        def deserialize(cls, data: StaticData) -> Serializable:
            raise NotImplementedError

        def serialize(self) -> StaticData:
            raise NotImplementedError

    class Brain(Brain):
        def __init__(self, brain_dna):
            self.brain_dna = brain_dna

        def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
            parsed_coords = body.to_tree_coordinates()

            bounds = np.zeros((2, 3), dtype=int)
            np.quantile([c[1].tolist() for c in parsed_coords], [0, 1], axis=0, out=bounds)

            if bounds[0][2] != bounds[1][2]:
                raise NotImplementedError("Can only handle planar robots (with z=0 for all modules)")
            xrange = max(-bounds[0][0], bounds[1][0])
            yrange = max(-bounds[0][1], bounds[1][1])

            hinges_map = {
                c[0].id: (c[1].x / xrange, c[1].y / yrange) for c
                in parsed_coords if isinstance(c[0], ActiveHinge)
            }

            io = [([Point(p[0], p[1], -1), Point(p[0], p[1], 1)]) for did in dof_ids if (p := hinges_map[did])]
            assert all(len(set(io_)) == len(io_) for io_ in io)  # Ensure no duplicates
            inputs, outputs = np.array(io).T.tolist()

            return ANNControl.Controller(self.brain_dna, inputs, outputs)


class CPGControl:
    class Brain(BrainCpgNetworkNeighbour):
        def __init__(self, brain_dna):
            self.genome = brain_dna

        def _make_weights(
                self,
                active_hinges: List[ActiveHinge],
                connections: List[Tuple[ActiveHinge, ActiveHinge]],
                body: Body,
        ) -> Tuple[List[float], List[float]]:
            cppn = abrain.CPPN(self.genome)

            internal_weights = [
                cppn(Point(*pos), Point(*pos), CPPN.Output.Weight)
                for pos in [
                    body.grid_position(active_hinge) for active_hinge in active_hinges
                ]
            ]

            external_weights = [
                cppn(Point(*pos1), Point(*pos2), CPPN.Output.Weight)
                for (pos1, pos2) in [
                    (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                    for (active_hinge1, active_hinge2) in connections
                ]
            ]

            return internal_weights, external_weights
