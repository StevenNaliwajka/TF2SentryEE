import numpy as np

from src.CV.analyzer import Analyzer
from typing import Sequence
from TurretSRC.CVImplementations.opencv_dnn_detector import DnnDetector
import cv2


class DefaultAnalyzer(Analyzer):
    def __init__(self, model_path: str, config: str):
        self.MODEL: cv2.dnn.Net = cv2.dnn.readNetFromTensorflow(model=model_path, config=config)
        self.INPUT_SIZE = (320,320)
        self.INPUT_SCALE = [1.0 / 127.5]
        self.INPUT_MEAN = (127.5,127.5,127.5)
        self.detector = DnnDetector(self.MODEL,self.INPUT_SCALE,self.INPUT_SIZE,self.INPUT_MEAN)

    def run_idle(self, new_frame: np.ndarray) -> bool:
        return len(self.detector.find_bboxes(new_frame)) > 0

    def run_target(self, new_frame: np.ndarray) -> np.ndarray:
        results: list[Sequence[int]] = self.detector.find_bboxes(new_frame)
        if len(results) == 0:
            return np.array([])
        else:
            return np.array(self.detector.find_bboxes(new_frame)[0])
