from __future__ import annotations
from typing import Sequence, cast

import numpy as np
import depthai as dai

from TurretSRC.RuntimeConfigs.DepthVision.oak_d_pipeline_component import OakDPipelineComponent
from src.CV.detector import Detector
from TurretSRC.StereoCameras.oak_d_stereo_camera import OakD


class OakDDetector(Detector, OakDPipelineComponent):
    dependencies: list[OakDPipelineComponent] = [OakD]

    def __init__(self, model: dai.NNModelDescription | dai.NNArchive, hyperparams: dict = None) -> None:
        """
        Args:
            model (dai.NNModelDescription | dai.NNArchive): The detection model you want to use.
            (https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/neural_network)
            hyperparams (dict): A dictionary of hyperparameters for the detector. They are all optional.
                Available hyperparameters are:
                confidence_threshold (float): confidence threshold to keep bounding boxes.
        """
        self._model: dai.NNModelDescription | dai.NNArchive = model
        self._hyperparams: dict
        if hyperparams is None:
            self._hyperparams = {}
        else:
            self._hyperparams = hyperparams

        self.camera: OakD = cast(OakD, None)
        self.PERSON_LABEL: int = cast(int, None)
        self.detection_queue: dai.MessageQueue = cast(dai.MessageQueue, None)

    def initialize_pipelines(self, parents: dict[str, OakDPipelineComponent]) -> None:

        OAKD_NAME: str = "oak_d_base"
        if not isinstance(parents[OAKD_NAME], OakD):
            raise ValueError("OakD instance not found under key \'oak_d_base\'")

        self.camera: OakD = cast(OakD, parents[OAKD_NAME])

        detection_network: dai.node.DetectionNetwork = (self.camera.pipeline.create(dai.node.DetectionNetwork))
        detection_network.build(self.camera.rgb_cam, self._model)
        self.PERSON_LABEL: int = detection_network.getClasses().index("person")

        detection_network.setConfidenceThreshold(self._hyperparams.get("confidence_threshold", 0.7))

        self.detection_queue: dai.MessageQueue = detection_network.out.createOutputQueue()

    def find_bboxes(self, frame: np.ndarray) -> list[Sequence[int]]:
        if not self.camera.pipeline.isRunning():
            raise RuntimeError("You are trying to pull data from the oak-d camera when the pipeline is not running.")

        result: dai.ImgDetections = cast(dai.ImgDetections, self.detection_queue.get())
        if result is None:
            return []

        return [self._normalize_detection(frame.shape, det) for det in result.detections if
                det.label == self.PERSON_LABEL]

    def _normalize_detection(self, height_width: Sequence[int], detection: dai.ImgDetection) \
            -> tuple[int, int, int, int]:
        height, width = height_width
        return (
            round(detection.xmin * width),
            round(detection.ymin * height),
            round(detection.xmax * width),
            round(detection.ymax * height)
        )
