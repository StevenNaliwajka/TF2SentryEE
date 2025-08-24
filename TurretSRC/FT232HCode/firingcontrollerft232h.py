# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
import time

import board
import digitalio

from src.IO.firingcontroller import FiringController


class FiringControllerFT232H(FiringController):

    def __init__(self):
        # init Shooting pin
        self.shoot = digitalio.DigitalInOut(board.C0)
        self.shoot.direction = digitalio.Direction.OUTPUT
        # For the switches I bought, True = Closed, False = Open
        self.shoot.value = True
        print("Initialized: Firing Controller")

    def fire(self) -> None:
        # Shoots one shot
        # For the switches I bought, True = Closed, False = Open
        self.shoot.value = False
        time.sleep(.1)
        self.shoot.value = True
