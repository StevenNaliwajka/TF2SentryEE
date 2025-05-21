from __future__ import annotations
import cv2
from pathlib import Path
from abc import abstractmethod, ABC

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..opencv_stereo_matcher import OpenCVStereoMatcher


class StereoMatcherCalibrator(ABC):
    def __init__(self, left_image_path: Path, right_image_path: Path,
                 left_stereo_map_path: Path | str = Path(__file__).parent / "saved_results/camera_calib/left_stereo_map.npz",
                 right_stereo_map_path: Path | str = Path(__file__).parent / "saved_results/camera_calib/right_stereo_map.npz"
                 ) -> None:
        """
        You should never create this class as it's an abstract class.
        You WILL need to set self.stereo_algo in your child class.
        params:
            left_image_path: The left stereo image path
            right_image_path: The right stereo image path
            left_stereo_map_path: The left stereo map path
                (this will be used for rectifying (aligning) the left image with the right one)
            right_stereo_map_path: The right stereo map path
        """

        super().__init__(left_stereo_map_path, right_stereo_map_path)
        self.rectified_left: cv2.typing.MatLike = None
        self.rectified_right: cv2.typing.MatLike = None

        self.load_new_img_pair(left_image_path, right_image_path)

        self.stereo_matcher = OpenCVStereoMatcher(left_stereo_map_path, right_stereo_map_path)
    
    @abstractmethod
    def _save_params(self) -> None:
        """
        Protected method.
        This function should somehow save the hyperparameters that you have chosen.
        For GUI based calibration, this will probably be a callback.
        For a CLI based calibration, you can just use a normal function.
        You don't have to strictly follow the above rules, but you should make sure that the params are saved
        somehow with this function.
        """
        pass

    @abstractmethod
    def tune_disparity_params(self):
        """
        This function should be the driver for tuning the hyperparameters of whatever stereo matcher you have chosen.
        """
        pass

    def load_new_img_pair(self, left_image_path: Path, right_image_path: Path) -> None:
        """
        This function will load a new image pair to compute disparity on.
        This pair will be corrected for stereo to be horizontally aligned and saved because the image will not change
        frequently.
        params:
            left_image_path: The image path to the left stereo image.
            right_image_path: The image path to the right stereo image.
        """
        if not left_image_path:
            raise ValueError("Left Image path is Empty or None!")
        if not right_image_path:
            raise ValueError("Right image path is Empty or None!")

        self.rectified_left, self.rectified_right = self.stereo_matcher.rectify_stereo_pair(
            cv2.imread(str(left_image_path), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(right_image_path), cv2.IMREAD_GRAYSCALE)
        )
    


    
