from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from abc import abstractmethod

from TurretSRC.StereoCameras.stereo_matcher import StereoMatcher


class OpenCVStereoMatcher(StereoMatcher):
    """
    Abstract class. You should not initialize this class.
    !!! You WILL need to set self.stereo_algo in whatever child class you initialize !!!

    This class will wrap all the OpenCV stereo Matching algorithms as well as helper methods for them
    e.g. Stereo BM or Stereo SGBM and allow for CUDA initialization if your hardware supports it as well
    as support for getting hyperparams.

    """

    def __init__(self, 
                 left_stereo_map_path: Path = Path(__file__).parent / "StereoCalibration/saved_results/camera_calib/left_stereo_map.npz",
                 right_stereo_map_path: Path = Path(__file__).parent / "StereoCalibration/saved_results/camera_calib/right_stereo_map.npz"
                 ) -> None:
        super().__init__()
        left_stereo_map, right_stereo_map = self._check_left_and_right_maps(left_stereo_map_path, right_stereo_map_path)

        self._left_stereo_map: tuple[np.ndarray, np.ndarray] = left_stereo_map
        self._right_stereo_map: tuple[np.ndarray, np.ndarray] = right_stereo_map

        # These are the parameters that you better set before calling set_stereo_maps.
        # You should probably put these after your init call if you decide to extend this class.
        self._stereo_algo: cv2.StereoMatcher = None

    @abstractmethod
    def get_all_hyperparams(self) -> dict:
        pass

    def rectify_stereo_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rectified_left: np.ndarray = cv2.remap(left_image,
                                               *self._left_stereo_map,
                                               interpolation=cv2.INTER_LINEAR,  # Slow but we only compute once.
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)  # Sets default border value to 0 (blk)

        rectified_right: np.ndarray = cv2.remap(right_image,
                                                *self._right_stereo_map,
                                                interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
        return rectified_left, rectified_right

    def get_disparity_map(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:

        return self._stereo_algo.compute(*self.rectify_stereo_pair(left_image, right_image))

    def get_disparity_no_rectification(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """
        This extra function was only added to help SLIGHT performance gains in calibration without exposing the private
        stereo_algo.
        """
        return self._stereo_algo.compute(left_image, right_image)

    def get_depth_map(self, left_image: np.ndarray, right_image: np.ndarray, focal_length_x_px: float,
                      baseline_mm: float) -> np.ndarray:
        return (focal_length_x_px * baseline_mm) / self.get_disparity_map(left_image, right_image)


    def call_setter_by_snk_case(self, snake_case_function_to_call: str, value: int) -> None:
        """
        CAUTION!!!
        This method is potentially dangerous. It calls getattr and if allow for snake_case_function_to_call
        to be controlled by the user in ANY way, this could allow for arbitrary code execution potentially.
        params:
            snake_case_function_to_call the snake case version of the function to call.
            ex: num_disparities -> setNumDisparities
            This then gets called by the stereo matcher.
        """
        if self._stereo_algo is None:
            raise ValueError("stereo_algo was not set. If you extended the MatcherCalibrator class, you need"
                             " to make sure that you set this value! ")
        method_to_call: str = self._cvt_snk_to_open_cv_method(snake_case_function_to_call)
        if hasattr(self._stereo_algo, method_to_call):
            getattr(self._stereo_algo, method_to_call)(value)
        else:
            raise ValueError("Unknown method " + method_to_call + ". Perhaps your json file is misformatted?")

    def call_getter_by_snk_case(self, snake_case_function_to_call: str) -> int:
        """
        CAUTION!!!
        This method is potentially dangerous. It calls getattr and if allow for snake_case_function_to_call
        to be controlled by the user in ANY way, this could allow for arbitrary code execution potentially.
        params:
            snake_case_function_to_call the snake case version of the function to call.
            ex: num_disparities -> getNumDisparities
            This then gets called by the stereo matcher.
        returns:
            the integer returned back by the getter called.
        """
        if self._stereo_algo is None:
            raise ValueError("stereo_algo was not set. If you extended the MatcherCalibrator class, you need"
                             " to make sure that you set this value! ")
        method_to_call: str = self._cvt_snk_to_open_cv_method(snake_case_function_to_call, "get")
        if hasattr(self._stereo_algo, method_to_call):
            return getattr(self._stereo_algo, method_to_call)()
        else:
            raise ValueError("Unknown method " + method_to_call + ". Perhaps your json file is misformatted?")

    def _cvt_snk_to_open_cv_method(self, string: str, mode: str = "set") -> str:
        """
        This function is made for the openCV StereoMatchers.
        It converts a snake case string to camelcase with a prepended word either set or get.
        e.g. num_disparities -> getNumDisparities if mode="get"
             block_size -> setBlockSize if mode="set"
        params:
            string: The string you want to convert in snake case.
            mode: the word you want to prepend to the front of it. Either "set" or "get"
        returns:
            The string in camelcase with the mode in front of it as demonstrated in examples above.

        """
        parts: list[str] = string.split("_")
        if mode == "set":
            return "set" + "".join(word.capitalize() for word in parts)
        elif mode == "get":
            return "get" + "".join(word.capitalize() for word in parts)
        else:
            raise ValueError("Unknown mode. Use either set or get")

    def _check_left_and_right_maps(self,
                                   left_stereo_map_path: Path | str,
                                   right_stereo_map_path: Path | str
                                   ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        This function will load the left and right stereo maps from the path and apply a bunch of checks on them
        params:
            left_stereo_map_path: The path of the left stereo map
            right_stereo_map_path: The path of the right stereo map
        returns:
            a double containing the left stereo map and the right stereo map.
        """
        if not (left_stereo_map_path and right_stereo_map_path):
            raise ValueError("left_stereo_map_path or right_stereo_map_path is Falsy (empty or None). "
                             "Please check your values! \n"
                             "LeftStereo: " + str(left_stereo_map_path) + "\n" +
                             "RightStereo: " + str(right_stereo_map_path))

        if not isinstance(left_stereo_map_path, (Path, str)):
            raise ValueError("left_stereo_map_path is not a path nor a string. ")
        if not isinstance(right_stereo_map_path, (Path, str)):
            raise ValueError("right_stereo_map_path is not a path nor a string. ")

        if isinstance(left_stereo_map_path, str):
            left_stereo_map_path = Path(left_stereo_map_path)
        if isinstance(right_stereo_map_path, str):
            right_stereo_map_path = Path(right_stereo_map_path)

        if not (left_stereo_map_path.exists() or left_stereo_map_path.is_file()):
            print(left_stereo_map_path.resolve())
            raise ValueError("Left stereo map path does not exist or is not a file")
        if not (right_stereo_map_path.exists() or right_stereo_map_path.is_file()):
            raise ValueError("Right stereo map path does not exist or is not a file")

        left_stereo_map: tuple = tuple(np.load(left_stereo_map_path)[key] for key in ("map1", "map2"))
        right_stereo_map: tuple = tuple(np.load(right_stereo_map_path)[key] for key in ("map1", "map2"))

        if len(left_stereo_map) != 2 or len(right_stereo_map) != 2:
            raise ValueError("Left Stereo map or right stereo map did not unpack the right number of elements")
        if not (isinstance(left_stereo_map[0], np.ndarray) or isinstance(left_stereo_map[1], np.ndarray)):
            raise ValueError("Left Stereo map[0] or Left Stereo map[1] is not a numpy array!")
        if not (isinstance(right_stereo_map[0], np.ndarray) or isinstance(right_stereo_map[1], np.ndarray)):
            raise ValueError("Right Stereo map[0] or Right Stereo map[1] is not a numpy array!")

        # noinspection PyTypeChecker
        left_stereo_map: tuple[np.ndarray, np.ndarray] = left_stereo_map
        # noinspection PyTypeChecker
        right_stereo_map: tuple[np.ndarray, np.ndarray] = right_stereo_map

        if not (left_stereo_map[0].dtype == np.int16 or left_stereo_map[1].dtype == np.uint16):
            raise ValueError("Left Stereo map has wrong type")
        if not (right_stereo_map[0].dtype == np.int16 or right_stereo_map[1].dtype == np.uint16):
            raise ValueError("Right Stereo map has wrong type")
        return left_stereo_map, right_stereo_map
