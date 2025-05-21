from __future__ import annotations
import dearpygui.dearpygui as dpg
import cv2.typing

from typing import Callable

import numpy as np


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DPGGUI(metaclass=Singleton):
    """
    This singleton class will manage all the GUI aspects of the Stereo Calibration.
    """
    def __init__(self,
                 slider_initialization: list[dict],
                 save_callback: Callable[[], None] = None,
                 slider_callback: Callable[[str, int], None] = (lambda *a, **k: None)
                 ) -> None:
        """
        Save callback is the function that should be called when the save button gets clicked.
        Slider callback is the function that should be called when a slider gets updated.
        Default will be a nop
        """
        self.PARAM_LIST: list = []
        for item in slider_initialization:
            self.PARAM_LIST.append(self._SliderParams(item))

        if save_callback:
            self.save_callback = save_callback
        dpg.create_context()
        self._slider_callback = slider_callback

    class _SliderParams:
        name: str
        tag: str
        default_value: int
        tooltip: str
        min_value: int
        max_value: int
        step: int

        def __init__(self, input_dict: dict):
            self.name = input_dict["name"]
            self.tag = input_dict["tag"]
            self.default_value = input_dict["default_val"]
            self.tooltip = input_dict["tooltip"]
            self.min_value = input_dict["min_val"]
            self.max_value = input_dict["max_val"]
            self.step = input_dict["step"]

    @staticmethod
    def _resize_primary_window() -> None:
        """
        This function should be called whenever the primary window is resized.
        This function will resize all the elements inside the viewport to scale accordingly.
        """
        x, y = dpg.get_item_rect_size("primary_window")
        dpg.set_item_height("current_disparity", y * 0.6)
        dpg.set_item_width("current_disparity", x * 0.8)
        dpg.set_item_width("spacer_l", x * 0.1)
        dpg.set_item_width("spacer_r", x * 0.1)

    def _update_param(self, sender_tag: str, real_val: int, user_params: _SliderParams) -> None:
        """
        This function should be called whenever the slider is updated.
        This function will update the slider to have the correct corresponding value.
        This will take into account the step parameter provided by modifying the real_val to correspond to
        the correct value.
        val = default + (step * real_val).
        For example: if default = 5, step = 2, real_val = -1, then val = 5 + (-1 * 2) = 3
        params:
            sender: the tag of the sender slider
            real_val: the real value of the slider (not the observed value but real value.)
                    e.g. slider may say "5" but in reality the value is 0, and we just shifted using the formula above.
            user_params:
                A struct containing a bunch of parameters.
                Please read the class description for what each parameter means.
                (Not to be rude, but so I don't have to maintain docs in 2 places.)
        """
        if sender_tag and user_params:
            val: int = user_params.default_value + user_params.step * real_val
            # dpg.set_value(sender, val)
            dpg.configure_item(sender_tag, format=val)
            self._slider_callback(sender_tag, val)

    def _create_slider_from_params(self, params: _SliderParams) -> None:
        """
        This function will create a bunch of integer sliders.
        Since there is no step functionality, this functionality is implemented by using the real value.
        What I mean by this while the slider says "5", the real value may be 0 and the slider computes the value using
        the formula as described in update_param. (val = default + (step * real_val))
        params:
            params: The slider params object containing all the information about this slider.
                You probably want to pull this from a json file (or you could hardcode it, I guess).
        """

        dpg.add_slider_int(label=params.name, tag=params.tag,
                           default_value=0,
                           min_value=(params.min_value - params.default_value) // params.step,
                           max_value=(params.max_value - params.default_value) // params.step,
                           callback=self._update_param,
                           user_data=params,
                           clamped=True,
                           format=str(params.default_value))
        with dpg.tooltip(params.tag):
            dpg.add_text(params.tooltip)

    def update_image(self, new_image: np.ndarray) -> None:
        assert new_image

        val: np.ndarray = dpg.get_value("disparity_img")
        val[:] = new_image

    def get_slider_val(self, tag: str) -> int:
        return dpg.get_value(tag)

    def show_gui(self, initial_img: np.ndarray, width: int, height: int) -> None:
        """
        This function does most of the heavy lifting in creating the GUI.

        initial_img better be a pointer to an image array of format RGBA, flattened and in [0,1] range.
        This invariant better be true because we are not checking it in the texture.
        """

        assert initial_img.dtype == np.float32

        with dpg.texture_registry(show=False):
            # noinspection PyTypeChecker
            dpg.add_raw_texture(width=width, height=height, default_value=initial_img, format=dpg.mvFormat_Float_rgba,
                                tag="disparity_img")

        with dpg.window(tag="primary_window"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True)
                dpg.add_table_column()
                dpg.add_table_column(width_fixed=True)
                with dpg.table_row():
                    dpg.add_spacer(tag="spacer_l")
                    dpg.add_image("disparity_img", tag="current_disparity")
                    dpg.add_spacer(tag="spacer_r")
            for param_tup in self.PARAM_LIST:
                self._create_slider_from_params(param_tup)

            dpg.add_button(label="Save config", tag="save_button", callback=self.save_callback)

        dpg.set_primary_window("primary_window", True)

        with dpg.item_handler_registry() as registry:
            dpg.add_item_resize_handler(callback=DPGGUI._resize_primary_window)
        dpg.bind_item_handler_registry("primary_window", registry)

        dpg.create_viewport(width=1024, height=728, title="Stereo Block Matching Calibration")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()

    def update_disparity_image(self, image: cv2.typing.MatLike) -> None:
        """
        This function will update the image at the top of the image.
        Said image should be in RGBA float format e.g. the pixels should be in range [0,1].
        """

        dpg.set_value("disparity_img", image)

    def __del__(self):
        dpg.destroy_context()
