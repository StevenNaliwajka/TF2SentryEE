from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class StereoMatcher(ABC):
    @abstractmethod
    def get_disparity_map(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """
        This function computes the depth map in mm.
        params:
            left_image: the left stereo image in greyscale.
            right_image: the right stereo image in greyscale.
            focal_length_x_px: the x direction focal length of both of the cameras in pixels.
            baseline_mm: the baseline length between the two stereo cameras in mm.
        returns:
            a depth map in mm.
        """
        pass

    @abstractmethod
    def rectify_stereo_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This function will apply stereo rectification on the image pairs.
        params:
            left_image: the left stereo image in greyscale
            right_image: the right stereo image in greyscale
        return:
            a double containing the rectified pair in the format (left, right)
        """
        pass

    @abstractmethod
    def get_depth_map(self, left_image: np.ndarray, right_image: np.ndarray, focal_length_x_px: float,
                      baseline_mm: float) -> np.ndarray:
        """
        This function computes the depth map in mm.
        params:
            left_image: the left stereo image in greyscale.
            right_image: the right stereo image in greyscale.
            focal_length_x_px: the x direction focal length of both of the cameras in pixels.
            baseline_mm: the baseline length between the two stereo cameras in mm.
        returns:
            a depth map in mm.
        """
        pass
