from __future__ import annotations
from abc import abstractmethod
from typing import Protocol


class OakDPipelineComponent(Protocol):
    """
    """

    @abstractmethod
    def initialize_pipelines(self, parents: dict[str, OakDPipelineComponent]) -> None:
        """
        This function should create OakD nodes and link them together. You should not do any linking
        inside __init__ as there is no guarantee that your upstream dependencies are ready.
        """
        pass
