import cv2
from IO.camera import Camera


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
    def __init__(self, resolution=(640,480), cap_fps=30) -> None:
        super().__init__(resolution, cap_fps)

        self.cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1])

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