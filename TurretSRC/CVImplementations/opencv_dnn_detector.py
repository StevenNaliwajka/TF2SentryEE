from __future__ import annotations

from typing import Sequence
import cv2
import numpy as np
import enum

from src.CV.detector import Detector


class DnnDetector(Detector):

    def __init__(self, network: cv2.dnn.Net, input_scale: Sequence[float] | float, input_size: tuple[int, int],
                 input_mean: Sequence[float] | float, nms_thresh: float = 0.5, conf_thresh: float = 0.5) -> None:
        """
        Note that this approach is limited to CPU usage only. If you want to use a GPU, you should use
        tensorRT, onnx, tensorflow, etc.
        Args:
            network (cv2.dnn.Net): The neural network converted to openCV format. You can use one of the many opencv
                helper functions to convert from tensorflow, onnx, etc to tensorflow.
            input_scale: how much the image should be scaled before being passed to nnet
            input_size: the size of the image to be passed to the nnet
            input_mean: the amount that should be removed from each channel before passing it through
                This is useful when you know that the dnn has been trained on normalized data.
            nms_thresh: how much overlap is allowed between bounding boxes. 1 = keep all, 0 = no overlap
            conf_thresh: how confident do we have to be to keep the bounding box.
        """
        self.conf_thresh: float = conf_thresh
        self.nms_thresh: float = nms_thresh
        self.input_size: tuple[int, int] = input_size
        self.input_scale: Sequence[float] | float = input_scale
        self.input_mean: Sequence[float] | float = input_mean

        if not 0 < self.nms_thresh < 1:
            raise ValueError("Your non maximum suppression must between (0,1)")
        self._net: cv2.dnn.DetectionModel = cv2.dnn.DetectionModel(network=network)

        self._net.setInputSize(self.input_size)
        self._net.setInputScale(self.input_scale)
        self._net.setInputMean(self.input_mean)
        self._net.setInputSwapRB(True)

    def find_bboxes(self, frame: np.ndarray) -> list[Sequence[int]]:
        """
        This function will run one iteration of the detection algorithm chosen.
        This function will return a list of bounding boxes of detected people.
        These bounding boxes will be represented as a quadruple in the form
        (startX,startY,endX,endY).
        If this list is empty, that means that the detection turned up nothing.
        """
        bbox_list: list[Sequence[int]] = []
        classIds, confs, bbox = self._net.detect(frame, confThreshold=self.conf_thresh, nmsThreshold=self.nms_thresh)
        print(classIds, confs, bbox)
        for i in range(len(bbox)):
            if classIds[i] == 1:
                bbox_list.append(bbox[i])
        return bbox_list
