import copy
from dataclasses import dataclass, astuple
from random import Random
from typing import Optional

from abrain import Genome as ANNGenome
from abrain.core.genome import GIDManager


@dataclass
class VisionData:
    w: int = 3
    h: int = 2

    def __iter__(self):
        return iter(astuple(self))


class RVGenome:
    __private_key = object()

    def __init__(self, brain: ANNGenome, vision: Optional[VisionData] = None):
        self.brain = brain
        self.vision = vision

    def with_vision(self):
        return self.vision is not None

    def id(self):
        return self.brain.id()

    def parents(self):
        return self.brain.parents()

    def __repr__(self):
        str_ = ""
        if self.with_vision():
            str_ += str(self.vision) + "-"
        return str_ + str(self.brain)

    def mutate(self, rng: Random) -> None:
        self.brain.mutate(rng)  ## TODO Add mutable vision

    def mutated(self, rng: Random, id_manager: GIDManager):
        return RVGenome(self.brain.mutated(rng, id_manager), self.vision)

    @staticmethod
    def random(rng: Random, id_manager: GIDManager) -> 'RVGenome':
        return RVGenome(
            ANNGenome.random(rng, id_manager),
            VisionData.random(rng)
        )

    def copy(self) -> 'Genome':
        return RVGenome(
            self.brain.copy(),
            copy.deepcopy(self.vision)
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, _):
        return self.copy()

    def __getstate__(self):
        return dict(brain=self.brain.__getstate__(), vision=self.vision.__dict__)

    def __setstate__(self, state):
        self.brain.__setstate__(state["brain"])
        self.vision.__dict__.update(state["vision"])

    def to_json(self):
        return dict(brain=self.brain.to_json(), vision=self.vision.__dict__)

    @staticmethod
    def from_json(data) -> 'RVGenome':
        """Recreate a RVGenome from string json representation
        """

        ## TODO Remove retro-compatibility
        if "brain" not in data:
            return RVGenome(ANNGenome.from_json(data), VisionData())

        return RVGenome(ANNGenome.from_json(data["brain"]),
                        VisionData(**data["vision"]))

    @staticmethod
    def from_dot(path: str, rng: Random):
        """Does not make sense for embedded abrain genome"""
        raise RuntimeError
