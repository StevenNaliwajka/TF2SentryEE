from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.CV.tracker import Tracker


class TrackerConfigurable(ABC):

    @abstractmethod
    def get_tracker(self) -> Tracker:
        pass
