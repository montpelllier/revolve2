from abc import ABC, abstractmethod

from ._batch import Batch
from ..scene import SimulationState


class Simulator(ABC):
    """Interface for a simulator."""

    @abstractmethod
    def simulate_batch(self, batch: Batch, vr, connection, onResultHandler) -> list[list[SimulationState]]:
        """
        Simulate the provided batch by simulating each contained scene.

        :param batch: The batch to run.
        :param vr:
        :param connection:
        :param onResultHandler:
        :returns: List of simulation states in ascending order of time.
        """
        pass
