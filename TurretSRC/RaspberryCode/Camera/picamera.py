from picamera2 import Picamera2
from IO.camera import Camera
import cv2
import numpy as np

class PiCamera(Camera):
    def __init__(self, resolution=(640,480), cap_fps=30) -> None:
        super().__init__(resolution, cap_fps)
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(main={"format": 'YUV420',"size": (resolution[0], resolution[1])})
        self.camera.configure(config)
        self.camera.start()
        self.raw_buffer:memoryview = None
    

    def get_frame(self) -> None:
        self.raw_buffer = self.camera.capture_buffer()

    def decode_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        if self.raw_buffer is None:
            return False, np.array([])

        raw_data = np.frombuffer(self.raw_buffer,dtype=np.uint8)
        return True, raw_data.reshape((self.RESOLUTION[1] * 3 // 2, self.RESOLUTION[0]))

    