from __future__ import annotations
from typing import Sequence, cast

import numpy as np
import depthai as dai

from TurretSRC.StereoCameras.oak_d_pipeline_component import OakDPipelineComponent
from src.CV.detector import Detector
from TurretSRC.StereoCameras.oak_d_stereo_camera import OakD


class OakDDetector(Detector, OakDPipelineComponent):
    dependencies: list[OakDPipelineComponent] = [OakD]

    def __init__(self, oakd_camera: OakD, model: dai.NNModelDescription | dai.NNArchive, hyperparams: dict) -> None:
        self.camera: OakD = oakd_camera
        detection_network: dai.node.DetectionNetwork = (self.camera.pipeline.create(dai.node.DetectionNetwork))
        detection_network.build(self.camera.rgb_cam, model)
        self.PERSON_LABEL: int = detection_network.getClasses().index("person")

        detection_network.setConfidenceThreshold(hyperparams.get("confidence_threshold", 0.7))

        self.detection_queue: dai.MessageQueue = detection_network.out.createOutputQueue()

    def initialize_pipelines(self, parents: list[OakDPipelineComponent]) -> None:
        oakd: OakD = None
        for node in parents:
            if isinstance(node, OakD):
                oakd = node
        if oakd is None:
            raise ValueError("OakD was not passed as a parent")

    def find_bboxes(self, frame: np.ndarray) -> list[Sequence[int, int, int, int]]:
        if not self.camera.pipeline.isRunning():
            raise RuntimeError("You are trying to pull data from the oak-d camera when the pipeline is not running.")

        result: dai.ImgDetections = cast(dai.ImgDetections, self.detection_queue.get())
        if result is None:
            return []

        return [self._normalize_detection(frame.shape, det) for det in result.detections if
                det.label == self.PERSON_LABEL]

    def _normalize_detection(self, height_width: Sequence[int, int], detection: dai.ImgDetection) \
            -> tuple[int, int, int, int]:
        height, width = height_width
        return (
            round(detection.xmin * width),
            round(detection.ymin * height),
            round(detection.xmax * width),
            round(detection.ymax * height)
        )
