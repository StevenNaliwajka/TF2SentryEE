from __future__ import annotations

import threading

import cv2
from IO.camera import Camera
from threading import Lock

'''
Arducam Resolutions
3264/2448	4:3     (15 fps)
2592/1944	4:3     (15 fps)
1920/1080	16:9    (30 fps)
1600/1200	4:3     (30 fps)
1280/720 	16:9    (30 fps)
800/600		4:3     (30 fps)
640/480		4:3     (30 fps)
320/240 	4:3     (30 fps)
'''


class Arducam(Camera):
    def __init__(self, camera_num: int, resolution=(640, 480), cap_fps=30,
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

        if 'GStreamer: YES' in cv2.getBuildInformation():
            self.cap = cv2.VideoCapture(camera_num, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(camera_num)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, cap_fps)

        self._lock: threading.Lock = Lock()

        self.print_info()

    def print_info(self) -> None:
        print("-------INFO--------")
        print("Using resolution:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Using backend:", self.cap.getBackendName())
        print("FPS: ", self.cap.get(cv2.CAP_PROP_FPS))

    def grab_frame(self) -> None:
        self.cap.grab()

    def decode_frame(self) -> bool:
        success, frame = self.cap.retrieve()
        if success:
            self._queue.push(frame)
        else:
            return False

    def consume_frame(self) -> cv2.typing.MatLike | None:
        """
        Returns the frame that the camera is currently reading.
        returns img:MatLike row,col,BGR24 (8 bits per color) or None if the queue is empty
        """
        return self._queue.pop()

    def release_camera(self) -> None:
        with self._lock:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except RuntimeError as e:
                print("Unable to release arducam", e)
