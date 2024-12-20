import cv2
from IO.camera import Camera

class DefaultCamera(Camera):
    def __init__(self, resolution=(640,480), cap_fps=30) -> None:
        super().__init__(resolution, cap_fps)

        # Dont worry about resizing image because
        # we're getting the correct res from the source.
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.CAPTURE_FPS)
        # IT IS VERY VERY IMPORTANT THAT IT'S YU12
        # the camera will give bad image if not configured this way.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('I', '4', '2', '0'))

        self.print_info()
    
    def print_info(self) -> None:
        print("-------INFO--------")
        print("Using resolution:",self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),"x",self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Using backend:",self.cap.getBackendName())
        print("FPS: ",self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame(self) -> None:
        self.cap.grab()

    def decode_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        """
        Returns the frame that the camera is currently reading.
        returns img:MatLike row,col,BGR24 (8 bits per color)
        """
        return self.cap.retrieve()
