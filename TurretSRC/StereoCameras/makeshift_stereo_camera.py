from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import cv2

import numpy as np

if TYPE_CHECKING:
    from IOImplementations.TurretSRC.StereoCameras.stereo_matcher import StereoMatcher
    from IO.stereo_camera import StereoCamera
from concurrent.futures import ThreadPoolExecutor, Future

from IO.camera import Camera


class MakeshiftStereoCamera(StereoCamera):
    LEFT_STEREO_DIRECTORY: str = "stereo_l"
    RIGHT_STEREO_DIRECTORY: str = "stereo_r"

    OBJ_PTS_FILENAME: str = "obj_pts.npz"
    LEFT_RES_FILENAME: str = "left_cam_calib_results.npz"
    RIGHT_RES_FILENAME: str = "right_cam_calib_results.npz"

    STEREO_PARAMS_PATH: str = "params_path.json"

    def __init__(self,
                 left_camera: Camera,
                 right_camera: Camera,
                 stereo_matcher: StereoMatcher,
                 baseline_length_mm: Optional[float] = None,
                 focal_length_px: Optional[np.ndarray] = None,
                 ) -> None:
        """
        left_camera: Camera a reference to the left camera
        right_camera: Camera a reference to the right camera
        matcher_choice: A value to which enum the user wishes to use.
        These 2 cameras should be parallel (or have very small angle between them) and should be the same camera.
        !!
        e.g. have about the same focal length. These parameters WILL be assumed true for calculations later.
        If you are not following these assumptions, then you must implement your own stereo class
        !!
        stereo_strategy: StereoCalibrator what stereo correspondence algorithm we should use.
        By default, this will be Block Matching (StereoBMCalibrator). This is quick but not the most accurate.
        """
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
        self._left_camera: Camera = left_camera
        self._right_camera: Camera = right_camera
        self._stereo_matcher: StereoMatcher = stereo_matcher

        self.baseline_mm: float = baseline_length_mm
        self.focal_length_px: np.ndarray = focal_length_px

    def get_images(self, timeout_secs: float = 1e-1) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        This function assumes that the cameras are parallel to each other AND that the cameras share at lest a little
        bit of the frame with each other.
        This function will get both the stereo images at once.
        timeout_secs: float the number of seconds before raising an exception.
            (This is necessary because we don't want desync between the cameras)
        """
        left_success: Future = self._executor.submit(self._left_camera.get_frame)
        right_success: Future = self._executor.submit(self._right_camera.get_frame)

        left_success.result(timeout_secs)
        right_success.result(timeout_secs)
        left_camera_result: Future = self._executor.submit(self._left_camera.decode_frame)
        right_camera_result: Future = self._executor.submit(self._right_camera.decode_frame)

        left_res_tuple: tuple[bool, cv2.typing.MatLike] = left_camera_result.result(timeout_secs)
        right_res_tuple: tuple[bool, cv2.typing.MatLike] = right_camera_result.result(timeout_secs)

        if not (left_res_tuple[0] and right_res_tuple[0]):
            if not left_res_tuple[0]:
                raise RuntimeError("Left camera failed to grab a new frame")
            else:
                raise RuntimeError("Right camera failed to grab a new frame")

        return left_res_tuple[1], right_res_tuple[1]

    def get_image_with_depth(self) -> cv2.typing.MatLike:
        """
        This function gets the image as well as the depth of each pixel in the image.
        returns:
            This function returns a 4 channel matrix (H,W,4) where the first 3 channels are the RGB and the 4th channel
            is the depth in mm.
        """
        if self.baseline_mm is None:
            raise ValueError("Your baseline is None! I cannot compute depth without the baseline \n"
                             "If you left the baseline unfilled in the constructor, did you make sure to call the function"
                             "`set_baseline_and_focal_length_px` later??")
        if not self.focal_length_px:
            raise ValueError("Your focal length is None/Empty! I cannot compute depth without the x focal length in pixels \n"
                             "If you left the focal length unfilled in the constructor, did you make sure to call the function"
                             "`set_baseline_and_focal_length_px` later??")
        left_image, right_image = self.get_images()

        depth_map: np.ndarray = self._stereo_matcher.get_depth_map(
            cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY),
            float(self.focal_length_px[0]),
            self.baseline_mm
        )
        return np.dstack((left_image, depth_map))
    
    def set_baseline_and_focal_length_px(self,baseline_mm: float, focal_length_px: np.ndarray) -> None:
        """
        This function allows for delayed initialization of the baseline and focal length.
        The reason that this is useful is because we may use the computed values for the baseline and
        focal length instead of theoretical values.
        """
        self.baseline_mm = baseline_mm
        self.focal_length_px = focal_length_px
    
    def get_baseline(self) -> float:
        return self.baseline_mm
    
    def get_focal_length_px(self) -> np.ndarray:
        return self.focal_length_px
