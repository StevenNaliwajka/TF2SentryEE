from __future__ import annotations
import cv2
import json

from pathlib import Path
from helper_functions import has_compiled_with_cuda

import numpy as np
from .opencv_stereo_matcher import OpenCVStereoMatcher


class StereoBM(OpenCVStereoMatcher):
    BM_PARAMS_JSON: Path = Path(__file__).parent / "StereoCalibration/saved_results/hyperparams/stereo_bm_default_values.json"

    HYPERPARAM_NAMES: set = {
        "num_disparities", "block_size", "pre_filter_size", "pre_filter_cap", "texture_threshold", "uniqueness_ratio",
        "speckle_range", "speckle_window_size", "disp12_max_diff", "min_disparity"
    }

    def __init__(self,
                 left_stereo_map_path: Path = Path(__file__).parent / "StereoCalibration/saved_results/camera_calib/left_stereo_map.npz",
                 right_stereo_map_path: Path = Path(__file__).parent / "StereoCalibration/saved_results/camera_calib/right_stereo_map.npz",
                 preexisting_params_path: Path | None = BM_PARAMS_JSON
                 ) -> None:
        super().__init__(left_stereo_map_path, right_stereo_map_path)

        left_stereo_map, right_stereo_map = self._check_left_and_right_maps(left_stereo_map_path, right_stereo_map_path)

        self._left_stereo_map: tuple[np.ndarray, np.ndarray] = left_stereo_map
        self._right_stereo_map: tuple[np.ndarray, np.ndarray] = right_stereo_map

        if has_compiled_with_cuda() and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # You will see a linter warning here if you do not have the CUDA compiled version of openCV
            # on your IDE.
            self._stereo_algo: cv2.cuda.StereoBM = cv2.cuda.createStereoBM()
        else:
            self._stereo_algo: cv2.StereoBM = cv2.StereoBM.create()

        if preexisting_params_path is None:
            self.initialize_hyperparams_from_json(self.BM_PARAMS_JSON)
        else:
            self.initialize_hyperparams_from_json(preexisting_params_path)
        
    def initialize_hyperparams_from_json(self, json_path: Path) -> None:
        """
        Your JSON file is required to be a list of objects of which have the attributes "name" and "default_val"
        for each hyperparameter in HYPERPARAM_NAMES (defined at the top of this file).
        params:
            json_path the path of the json_path that you're going to initialize your hyperparameters from
            (the default values)
        """

        if not json_path:
            raise ValueError("Json path is either empty, nonexistent, or None")

        with open(json_path) as file:
            param_list: list = json.load(file)
            if {param["name"] for param in param_list} != self.HYPERPARAM_NAMES:
                raise ValueError("You are missing some of the required hyperparameters in your saved file."
                                 "If you havent set all of them, you probably should just pass None to initialize them"
                                 "to their default values and then tune them from there.")
            for item in param_list:
                if not isinstance(item["default_val"], int):
                    raise ValueError("Your default_value for " + str(item["name"]) + "is not an integer!")
                # Be careful! While this saves us the hassle of massive if-else chains, this allows for
                # attackers to maybe call whatever method they want.
                # make sure you're not pulling your json from untrusted sources.
                self.call_setter_by_snk_case(item["name"], item["default_val"])

    def get_all_hyperparams(self) -> dict:
        """
        This function will return all hyperparams.
        returns:
            A dict containing all hyperparam values in the format hyperparam_name: hyperparam_val
        """
        return {
            name: self.call_getter_by_snk_case(name)
            for name in self.HYPERPARAM_NAMES
        }
