from __future__ import annotations
from src.RuntimeConfigs.depth_vision_configurable import DepthVisionConfigurable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.IO.depthvision import DepthVision
    from TurretSRC.StereoCameras.StereoCalibration.stereo_calibrate import CameraCalibrationResults
    from TurretSRC.StereoCameras.StereoCalibration.stereo_calibrate import StereoCalibrationResults

import TurretSRC.StereoCameras.StereoCalibration.stereo_calibrate as calibrate
from pathlib import Path
from TurretSRC.StereoCameras.makeshift_stereo_camera import MakeshiftStereoCamera
from TurretSRC.FT232HCode.Camera.arducam import Arducam
from TurretSRC.StereoCameras.stereo_matcher import StereoMatcher
from TurretSRC.StereoCameras.stereo_bm import StereoBM
import numpy as np


class MakeshiftStereoConfig(DepthVisionConfigurable):
    STEREO_DIR: Path = Path("TurretSRC/StereoCameras/")

    def __init__(
            self,
            left_stereo_map: Path = STEREO_DIR / "saved_results/camera_calib/left_stereo_map.npz",
            right_stereo_map: Path = STEREO_DIR / "saved_results/camera_calib/right_stereo_map.npz",
            hyperparams_path: Path = STEREO_DIR / "StereoCalibration/saved_results/hyperparams/slider_bm_params.json"
    ) -> None:
        self.left_stereo_map: Path = left_stereo_map
        self.right_stereo_map: Path = right_stereo_map
        self.hyperparams_path: Path = hyperparams_path

    def get_depthvision(self) -> DepthVision:
        STEREO_MATCHER: StereoMatcher = StereoBM(self.left_stereo_map, self.right_stereo_map, self.hyperparams_path)

        results: dict = calibrate.load_requested_results(
            {"left_camera_info", "right_camera_info", "general_stereo_info_path"})
        left_camera_res: CameraCalibrationResults = results["left_camera_info"]
        right_camera_res: CameraCalibrationResults = results["right_camera_info"]
        general_stereo_info: StereoCalibrationResults = results["general_stereo_info_path"]

        # In the format (fx,fy)
        LEFT_FOCAL_LENGTH_PX = np.array(
            [left_camera_res.new_intrinsic_matrix[0][0], left_camera_res.new_intrinsic_matrix[1][1]])
        RIGHT_FOCAL_LENGTH_PX = np.array(
            [right_camera_res.new_intrinsic_matrix[0][0], right_camera_res.new_intrinsic_matrix[1][1]])

        print("Computed left focal length is", LEFT_FOCAL_LENGTH_PX)
        print("Computed right focal length is", RIGHT_FOCAL_LENGTH_PX)

        ABS_LIMIT: int = 10
        if not np.allclose(LEFT_FOCAL_LENGTH_PX, RIGHT_FOCAL_LENGTH_PX, rtol=0, atol=ABS_LIMIT):
            raise ValueError(
                "Your left and right focal lengths have more than " + str(ABS_LIMIT) + " pixels of disparity!!!"
                                                                                       "Are you sure that your camera "
                                                                                       "system is perfectly parallel?"
                                                                                       "If so, maybe try recalibrating."
            )
        return MakeshiftStereoCamera(Arducam(0),
                                     Arducam(1),
                                     STEREO_MATCHER,
                                     np.linalg.norm(general_stereo_info.translation_vector),
                                     LEFT_FOCAL_LENGTH_PX
                                     )
