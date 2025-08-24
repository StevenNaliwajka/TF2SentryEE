# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
from __future__ import annotations
import board
import digitalio
from threading import Event
from src.IO.wranglercontroller import WranglerController
from TurretSRC.FT232HCode.motorcontrollerft232h import MotorControllerFT232H


class WranglerControllerFT232H(WranglerController):

    def __init__(self, motor_controller: MotorControllerFT232H,
                 wrangler_event: Event):
        super().__init__(motor_controller, wrangler_event)
        self.Motorcontroller = motor_controller
        #Create the pin for the wrangler to monitor.
        self.statusPin = digitalio.DigitalInOut(board.D7)
        self.statusPin.direction = digitalio.Direction.INPUT
        print("Initialized: Wrangler Controller")
        self.turnstate = Event()
    #Checks the pin and updates event for the brain
    #if pin is low for more than 2 seconds, wrangler
    #is turned off.
    def check_status(self) -> None:
        if self.statusPin.value:
            self.wrangler_event.set()
        else:
            self.wrangler_event.clear()

    def increase_y_angle(self) -> None:
        # Sets the current angle. Max is MAX_Y_ANGLE DEG
        self.Motorcontroller.set_y_max()
        self.check_joystick_state()
        self.turnstate.wait()
        self.Motorcontroller.pause_y()
        self.turnstate.clear()

    def decrease_y_angle(self) -> None:
        self.Motorcontroller.set_y_min()
        self.check_joystick_state()
        self.turnstate.wait()
        self.Motorcontroller.pause_y()
        self.turnstate.clear()

    def increase_x_angle(self) -> None:
        # Sets the current angle. Max is MAX_Y_ANGLE DEG
        self.Motorcontroller.set_x_max()
        self.check_joystick_state()
        self.turnstate.wait()
        self.Motorcontroller.pause_x()
        self.turnstate.clear()

    def decrease_x_angle(self) -> None:
        self.Motorcontroller.set_x_min()
        self.check_joystick_state()
        self.turnstate.wait()
        self.Motorcontroller.pause_x()
        self.turnstate.clear()

    def read_radio(self) -> tuple[bool, WranglerController.Direction]:
        self.Motorcontroller._update_position()

        if self.Motorcontroller.fire_state == 1:
            tempData = True
        else:
            tempData = False

        if self.Motorcontroller.move_state == 3:
            return tempData, WranglerController.Direction.LEFT
        elif self.Motorcontroller.move_state == 4:
            return tempData, WranglerController.Direction.RIGHT
        elif self.Motorcontroller.move_state == 1:
            return tempData, WranglerController.Direction.UP
        elif self.Motorcontroller.move_state == 2:
            return tempData, WranglerController.Direction.DOWN
        else:
            return tempData, WranglerController.Direction.NOP

    def check_joystick_state(self):
        joystickCount = 0
        complete = False
        while not complete:
            wranglerData = self.read_radio()
            # print("inside loop")
            if wranglerData[1] == WranglerController.Direction.NOP:
                joystickCount += 1
                if joystickCount >= 2:
                    self.turnstate.set()
                    complete = True
            else:
                joystickCount = 0

