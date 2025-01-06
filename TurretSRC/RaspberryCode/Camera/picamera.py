from picamera2 import Picamera2
from IO.camera import Camera
import cv2
import numpy as np


class PiCamera(Camera):
    def __init__(self, camera_num, resolution=(640, 480), cap_fps=30) -> None:
        super().__init__(resolution, cap_fps)
        self.camera = Picamera2(camera_num)
        # Note here that RGB888 is actually BGR with 8 bits for each color (as defined in the libcamera/picamera2 docs)
        # We will hardcode the sensor size here because we are using picamera module 2.
        config = self.camera.create_video_configuration(sensor={'output_size': (1640, 1232), 'bit_depth': 10},
                                                        main={"format": 'RGB888',
                                                              "size": (resolution[0], resolution[1])})
        self.camera.configure(config)
        self.camera.start()
        self.raw_frame: np.ndarray = None

    def get_frame(self) -> None:
        self.raw_frame = self.camera.capture_array()

    def decode_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        if self.raw_frame is None:
            return False, np.array([])

        return True, cv2.resize(self.raw_frame, self.RESOLUTION, interpolation=cv2.INTER_AREA)

    