from picamera2 import Picamera2
from IO.camera import Camera
import cv2
import numpy as np

class PiCamera(Camera):
    def __init__(self,camera_num,resolution=(640,480), cap_fps=30) -> None:
        super().__init__(resolution, cap_fps)
        self.camera = Picamera2(camera_num)
        # Note here that RGB888 is actually BGR with 8 bits for each color (as defined in the libcamera/picamera2 docs)
        config = self.camera.create_video_configuration(main={"format": 'RGB888',"size": (resolution[0], resolution[1])})
        self.camera.configure(config)
        self.camera.start()
        self.raw_buffer:memoryview = None
    

    def get_frame(self) -> None:
        self.raw_buffer = self.camera.capture_buffer()

    def decode_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        if self.raw_buffer is None:
            return False, np.array([])

        return True, np.reshape(np.frombuffer(self.raw_buffer,dtype=np.uint8),
                                (self.RESOLUTION[1],self.RESOLUTION[0],3))

    