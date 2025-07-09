import numpy as np

from src.CV.analyzer import Analyzer
from typing import Sequence
from src.IOImplementations.TurretSRC.CVImplementations.opencv_detector import PersonDetector


class DefaultAnalyzer(Analyzer):
    def __init__(self):
        self.detector = PersonDetector()

    def run_idle(self, new_frame: np.ndarray) -> bool:
        return len(self.detector.find_bboxes(new_frame)) > 0

    def run_target(self, new_frame: np.ndarray) -> np.ndarray:
        results: list[Sequence[int]] = self.detector.find_bboxes(new_frame)
        if len(results) == 0:
            return np.array([])
        else:
            return np.array(self.detector.find_bboxes(new_frame)[0])
