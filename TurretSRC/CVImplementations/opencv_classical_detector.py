from typing import Sequence
import cv2
import numpy as np
import enum

from src.CV.detector import Detector


class ClassicalDetector(Detector):
    class DetectorOptions(enum):
        """
        haar: if you want to use the naive openCV haar transform (its fast but inaccurate)
        hog: if you want to use the histogram of gradients method.
        """
        haar = 1
        hog = 2

    def __init__(self, strategy: DetectorOptions, hyperparams: dict) -> None:
        """
        Args:
            strategy (ClassicalDetector.DetectorOptions): The detection strategy you want to use.
            hyperparams (dict): A dictionary of key-value pairs of hyperparams. This will depend on your strategy.
                This will just be the inputParams to cv2.detectMultiScale for either cv2.CascadeClassifier() for haar
                or cv2.HOGDescriptor() for hog
        """
        self.hyperparams: dict = hyperparams
        if strategy == self.DetectorOptions.haar:
            self._model = cv2.CascadeClassifier()
        elif strategy == self.DetectorOptions.hog:
            self._model: cv2.HOGDescriptor = cv2.HOGDescriptor()
            self._model.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        else:
            raise ValueError("Invalid enum passed!")

    def find_bboxes(self, frame: np.ndarray) -> list[Sequence[int]]:
        """
        This function will run one iteration of the detection algorithm chosen.
        This function will return a list of bounding boxes of detected people.
        These bounding boxes will be represented as a quadruple in the form
        (startX,startY,endX,endY).
        If this list is empty, that means that the detection turned up nothing.
        """
        bbox_list, _ = self._model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), **self.hyperparams)

        return list(bbox_list)
