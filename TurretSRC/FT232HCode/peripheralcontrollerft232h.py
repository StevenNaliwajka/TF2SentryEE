# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
import digitalio
import board
from src.IO.peripheralcontroller import PeripheralController


class PeripheralControllerFT232H(PeripheralController):

    def __init__(self):
        # init LED Pin
        self._LED = digitalio.DigitalInOut(board.C1)
        self._LED.direction = digitalio.Direction.OUTPUT
        # For the switches I bought, True = Closed, False = Open
        self._LED.value = True
        print("Initialized: Peripheral Controller")

    def led_state(self, state: bool) -> None:
        # Updates LED state
        # For the switches I bought, True = Closed, False = Open
        # Accepts the users standard logic and inverts: of True = Light on, False = Light off
        self._LED.value = not state
