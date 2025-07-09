from typing import Sequence
import cv2

from src.CV.detector import Detector


class PersonDetector(Detector):
    # Percentage threshold, anything above this confidence
    # we should accept. Should be in between 0-1
    CONF_THRESH: float = 0.5
    NMS_THRESH: float = 0.5
    assert (0 <= CONF_THRESH <= 1)
    MODEL_PATH: str = "CVModel/Detection/frozen_inference_graph.pb"
    PBTEXT: str = "CVModel/Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    def __init__(self) -> None:
        self._model: cv2.dnn.Net = cv2.dnn.readNetFromTensorflow(model=self.MODEL_PATH, config=self.PBTEXT)
        self._net: cv2.dnn.DetectionModel = cv2.dnn.DetectionModel(network=self._model)

    def find_bboxes(self, frame) -> list[Sequence[int]]:
        """
        This function will run one iteration of the detection algorithm chosen.
        This function will return a list of bounding boxes of detected people.
        These bounding boxes will be represented as a quadruple in the form
        (startX,startY,endX,endY).
        If this list is empty, that means that the detection turned up nothing.
        """
        self._net.setInputSize(320, 320)
        self._net.setInputScale([1.0 / 127.5])  #TODO: confirm this didnt break the detection.
        self._net.setInputMean((127.5, 127.5, 127.5))
        self._net.setInputSwapRB(True)

        bbox_list = []
        classIds, confs, bbox = self._net.detect(frame, confThreshold=self.CONF_THRESH,
                                                 nmsThreshold=self.NMS_THRESH)
        print(classIds, confs, bbox)
        for i in range(len(bbox)):
            if classIds[i] == 1:
                bbox_list.append(bbox[i])
        return bbox_list
