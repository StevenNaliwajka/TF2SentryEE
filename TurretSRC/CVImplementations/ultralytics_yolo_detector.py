from __future__ import annotations

from typing import Sequence

from pathlib import Path
from ultralytics.engine.results import Results
from ultralytics import YOLO

import numpy as np

from src.CV.detector import Detector


class UltralyticsYoloDetector(Detector):

    def __init__(self, model_path: Path, hyperparams: dict) -> None:
        """
        List of acceptable hyperparameters:
        https://docs.ultralytics.com/modes/predict/#inference-arguments
        If you omit it, it will use the ultralytics default (which is often a good choice).

        Args:
            model_path (Path): The path of the model that you want to use in the correct format.
            hyperparams (dict): A dictionary containing the parameterises that you want to tweak for this model.
        """
        self.hyperparams: dict = hyperparams
        self._model: YOLO = YOLO(model=model_path, task="detection")

    def find_bboxes(self, frame: np.ndarray) -> list[Sequence[int]]:
        result_list: list[Results] = self._model.predict(frame, classes=[0], **self.hyperparams)

        return [result.boxes.xyxy.round().int() for result in result_list]
