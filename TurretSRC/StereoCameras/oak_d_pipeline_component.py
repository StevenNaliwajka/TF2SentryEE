from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type


class OakDPipelineComponent(ABC):

    dependencies: list[Type[OakDPipelineComponent]] = []
    # You will need to overload this depending on what dependencies you have.
    # Remember, no duplicates!

    @abstractmethod
    def initialize_pipelines(self, parents: list[OakDPipelineComponent]) -> None:
        pass
