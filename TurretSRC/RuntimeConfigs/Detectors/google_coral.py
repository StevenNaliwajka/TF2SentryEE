from __future__ import annotations
from src.RuntimeConfigs.depth_vision_configurable import DepthVisionConfigurable
from TurretSRC.CVImplementations.ultralytics_yolo_detector import UltralyticsYoloDetector
import enum
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import TurretSRC.RuntimeConfigs.Detectors.ultralytics_converter as converter

if TYPE_CHECKING:
    from src.CV.detector import Detector


class GoogleCoralDetector(DepthVisionConfigurable):
    class DeploymentStrategy(enum):
        ULTRALYTICS = 0

    def __init__(self,
                 ultralytics_model: str | Path = "yolov8n.pt",
                 model_save_path: Path = Path("TurretSRC/CVImplementations/CVModels/Detection"),
                 strategy: DeploymentStrategy = DeploymentStrategy.ULTRALYTICS
                 ) -> None:
        """
        Args:
            ultralytics_model: The string or the path of the detector that you want to use.
                Valid strings can be found on the ultralytics website.
                For YOLOV8: https://docs.ultralytics.com/models/yolov8/?utm_source=chatgpt.com#supported-tasks-and-modes
                Note that there are other valid strings, but ultralytics does not list them all in one place.
                e.g. yolov11 models etc.
            model_save_path: The path of where you want to the model.
            strategy: Currently only one option. Ultralytics. Reserved for the future for when alternative
                execution strategies exist. (e.g. pytorch)
        """
        self._ultralytics_model_key: str | Path = ultralytics_model
        self.model_save_path: Path = model_save_path
        # Save the cache directory so that when you initialize a model download, you save to the specified path
        # instead of saving to the home directory

        if strategy == GoogleCoralDetector.DeploymentStrategy.ULTRALYTICS:
            self._getter_to_call: Callable[[], Detector] = self._use_ultralytics
        else:
            raise ValueError("Unknown google coral runtime mode selected!!")

    def _use_ultralytics(self) -> UltralyticsYoloDetector:
        """
        You will need to install https://github.com/feranick/libedgetpu.
        Guide here: https://docs.ultralytics.com/guides/coral-edge-tpu-on-raspberry-pi/

        Returns:
            This function returns an instance of an UltralyticsYoloDetector running on the EdgeTPU
        """

        # This model may or may not be of the edgeTPU form already. So we will check and convert it if needed.
        save_path: Path = converter.convert_to_proper_format(self.model_save_path,
                                                             self._ultralytics_model_key, {"imgsz": 320})

        return UltralyticsYoloDetector(save_path, {})

    def get_depthvision(self) -> Detector:
        return self._use_ultralytics()
