from __future__ import annotations

import threading

from picamera2 import Picamera2
from IO.camera import Camera
import cv2
import numpy as np
from threading import Lock


class PiCamera(Camera):
    def __init__(self, camera_num, resolution=(640, 480), cap_fps=30,
                 capacity: int = 2, blocking: bool = False) -> None:
        """
        Parameters:
        cap_fps:int is the number of frames that we will take per second for our camera. (or aim to)
        resolution:tuple(int,int) is the current resolution that we are using. Different from original_res as
            that value is what we measured to find angle. Format is (width,height)
        capacity: the number of elements we should have in our image buffer.
        blocking: whether we should block upon reaching the max capacity and wait for a pop or if we should instead
            just replace the oldest element.
        """
        super().__init__(resolution, cap_fps, capacity, blocking)
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

    def grab_frame(self) -> None:
        """
        This function grabs the next frame and decodes it, but in order to adhere to the interface, we still have to
        call decode_frame in order to add it to the queue.
        The reason we do this is that the picamera has an ISP which is hardware accelerated, so it's faster
        to not decode on host CPU.
        """
        self._raw_frame = self.camera.capture_array()

    def decode_frame(self) -> bool:
        """
        Actually here, the frame we have is already good, so we just do a null check.
        """
        if self._raw_frame is None:
            return False

        self._queue.push(self._raw_frame)
        return True

    def consume_frame(self) -> cv2.typing.MatLike | None:
        return self._queue.pop()

    def release_camera(self) -> None:
        with self._lock:
            try:
                if self.camera.running:
                    self.camera.close()
            except RuntimeError as e:
                print("Error closing the picamera", e)
