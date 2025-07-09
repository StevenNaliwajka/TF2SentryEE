from __future__ import annotations

from picamera2 import Picamera2
from src.IO.camera import Camera
import cv2
import numpy as np
from threading import Lock

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.IOImplementations.TurretSRC.StereoCameras.StereoCalibration.stereo_calibrate import CameraCalibrationResults

class PiCamera(Camera):
    def __init__(self, camera_num, resolution=(640, 480), cap_fps=30,
                 camera_calib_results: CameraCalibrationResults = None,
                 capacity: int = 1, blocking: bool = False) -> None:
        """
        Parameters:
        cap_fps:int is the number of frames that we will take per second for our camera. (or aim to)
        resolution:tuple(int,int) is the current resolution that we are using. Different from original_res as
            that value is what we measured to find angle. Format is (width,height)
        capacity: the number of elements we should have in our image buffer.
        blocking: whether we should block upon reaching the max capacity and wait for a pop or if we should instead
            just replace the oldest element.
        """
        super().__init__(resolution, cap_fps, camera_calib_results, capacity, blocking)
        self.camera = Picamera2(camera_num)
        # Note here that RGB888 is actually BGR with 8 bits for each color (as defined in the libcamera/picamera2 docs)
        # We will hardcode the sensor size here because we are using picamera module 2.
        config = self.camera.create_video_configuration(sensor={'output_size': (1640, 1232), 'bit_depth': 10},
                                                        main={"format": 'RGB888',
                                                              "size": (resolution[0], resolution[1])})
        self.camera.configure(config)
        self.camera.start()
        self._raw_frame: np.ndarray = None
        self._lock: Lock = Lock()

    def get_frame(self) -> None:
        self._raw_frame = self.camera.capture_array()

    def decode_frame(self, undistort: bool = False) -> tuple[bool, cv2.typing.MatLike]:
        if self._raw_frame is None:
            return False, np.array([])

        if undistort:
            if self.calibration_info is None:
                raise ValueError("You never set the camera calibration info.")

            return cv2.undistort(self._raw_frame, self.calibration_info.old_intrinsic_matrix,
                                 self.calibration_info.distortion_coeffs, None,
                                 self.calibration_info.new_intrinsic_matrix)
        else:
            return True, self._raw_frame

    def release_camera(self) -> None:
        with self._lock:
            try:
                if self.camera.running:
                    self.camera.close()
            except RuntimeError as e:
                print("Error closing the picamera", e)
