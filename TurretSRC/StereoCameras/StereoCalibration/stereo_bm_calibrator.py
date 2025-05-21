from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from .dpg_gui import DPGGUI
import cv2
from .stereo_matcher_calibrator import StereoMatcherCalibrator

from ..stereo_bm import StereoBM


class StereoBMCalibrator(StereoMatcherCalibrator):
    SLIDER_BM_PARAMS_JSON: Path = Path(__file__).parent / "saved_results/hyperparams/slider_bm_params.json"

    def __init__(self,
                 left_image_path: Path,
                 right_image_path: Path,
                 left_stereo_map_path: Path | str = Path(__file__).parent / "saved_results/camera_calib/left_stereo_map.npz",
                 right_stereo_map_path: Path | str = Path(__file__).parent / "saved_results/camera_calib/right_stereo_map.npz",
                 new_save_path: Path = Path(__file__).parent / "saved_results/hyperparams/stereo_bm_params.json",
                 preexisting_params_path: Path | None = None
                 ) -> None:
        """
        params:
            left_image_path: the path to the left stereo image.
            right_image_path: the path to the right stereo image.
            left_stereo_map_path: the path to the left stereo map. Should be a tuple of np.ndarrays with the first
                element being 2-channel int16 and the second element being 1-channel uint16.
            right_stereo_map_path: the path to the right stereo map. Same format as left_stereo_map_path
            new_save_path: the path where we should save the hyperparameters that we have just tuned. Should be a
                json file path.
            preexisting_params_path: the path to whatever preexisting params if you want to tweak some params
                that you already have.
        """
        super().__init__(left_image_path, right_image_path, left_stereo_map_path, right_stereo_map_path)

        self.stereo_matcher: StereoBM = StereoBM(left_stereo_map_path, right_stereo_map_path, preexisting_params_path)

        self._save_path = new_save_path

        self.height: int = -1
        self.width: int = -1

        disparity, self.height, self.width = self.compute_disparity()

        self.curr_dpg_image: np.ndarray = self.convert_to_dpg_texture(disparity)

        self.dpg_gui: DPGGUI = DPGGUI(self._create_slider_dict(), self._save_params, self.on_slider_update)

    def _create_slider_dict(self) -> list[dict]:
        """
        This function will create the slider dictionary by merging the current StereoBM hyperparameters with
        the slider initialization information.
        returns:
            a list of dictionary's similar to json that contains all the information to initialize the sliders
            for DPG.
        """
        slider_info_list: list[dict] = []
        slider_bm_params: dict = self.stereo_matcher.get_all_hyperparams()
        with open(self.SLIDER_BM_PARAMS_JSON) as file:
            slider_params: list = json.load(file)
            if not isinstance(slider_params, list):
                raise ValueError("Your slider params json is not a list")
            for slider_info in slider_params:
                slider_info_list.append(slider_info["name"] | slider_bm_params["name"])
        return slider_info_list

    def tune_disparity_params(self) -> None:
        """
        This function starts the DPG gui and allows for the user to tune their hyperparameters.
        """
        self.dpg_gui.show_gui(self.curr_dpg_image, self.width, self.height)

    def _save_params(self) -> None:
        """
        This callback function will save the current hyperparameters that have been selected.
        You should bind this callback function to the save button on DPG.
        """
        hyperparams: dict = self.stereo_matcher.get_all_hyperparams()
        saved_param_list = [
            {
                "name": name,
                "default_val": hyperparams[name]
            }
            for name in hyperparams
        ]

        with open(self._save_path, "w") as file:
            json.dump(saved_param_list, file)

    def convert_to_dpg_texture(self, image: np.ndarray) -> np.ndarray:
        """
        This function will convert the image to DPG format to be used as a raw texture.
        You need to be very careful here in this function as DPG does not apply checks for raw textures,
        and you can easily segfault.
        params:
            image: A nxd image of RGBA format, float32. Both of these formats must be there otherwise you
                    will get image corruption at best and crash at worst.
        """
        if image.dtype != np.float32:
            raise ValueError("Your image datatype is incorrect!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        image = image.astype(np.float32)  # Important.
        image: np.ndarray = image / 255

        return image.flatten()

    def compute_disparity(self) -> tuple[np.ndarray, int, int]:
        """
        This function will compute disparity (with some extra checks because performance is not critical yet).
        It will return the disparity map after color is applied to it.
        Note that because the Stereo Block Matching algorithm is a meant to be primitive and run in real-time.
        It has a real problem with texture less/matching parts of the image as it cannot match with anything.
        If this is a problem with you, use a stronger (more computationally expensive algorithm).
        returns:
            The disparity map as a (h,w,3) BGR image,
            The height of the image,
            The width of the image.
        """
        if self.rectified_left is None or self.rectified_right is None:
            raise ValueError("Rectified left or rectified right is None! "
                             "Did you make sure to call load_stereo_maps() as well as load_image_pair() "
                             "in that order?")
        disparity: np.ndarray = self.stereo_matcher.get_disparity_map(self.rectified_left, self.rectified_right)
        disparity = (disparity.astype(np.float32) / 16) - self.stereo_matcher.call_getter_by_snk_case("min_disparity")
        disparity = disparity / self.stereo_matcher.call_getter_by_snk_case("num_disparities")
        disparity = np.clip(disparity, 0, 1)
        disparity = (disparity * 255).astype(np.uint8)
        height, width = disparity.shape[0], disparity.shape[1]
        return cv2.applyColorMap(disparity, cv2.COLORMAP_JET), height, width

    def on_slider_update(self, sender_tag: str, new_value: int) -> None:
        """
        This callback function should be bound to when a slider updates on DPG.
        It will pass in information about which slider changed and update the stereo block matcher accordingly.
        Afterward, it will compute the new disparity map and update the GUI respectively.
        params:
            sender_tag: The slider tag that sent the message. This will be defined in the json file above.
            new_value: The new value that the slider is taking on.
        """
        self.stereo_matcher.call_setter_by_snk_case(sender_tag, new_value)
        disparity, height, width = self.compute_disparity()
        self.height = height
        self.width = width
        # Note here that we never overwrite the memory address of the image.
        # This is because DPG requires us to just change the texture and not point to a new place.
        self.curr_dpg_image[:] = self.convert_to_dpg_texture(disparity)