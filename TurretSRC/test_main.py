# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
import time
from FT232HCode.motorcontrollerft232h import MotorControllerFT232H
from FT232HCode.firingcontrollerft232h import FiringControllerFT232H
from FT232HCode.peripheralcontrollerft232h import PeripheralControllerFT232H
from FT232HCode.soundcontrollerft232h import SoundControllerFT232H
from FT232HCode.wranglercontrollerft232h import WranglerControllerFT232H
from src.IO.wranglercontroller import WranglerController
from threading import Event
motor = MotorControllerFT232H()
fire = FiringControllerFT232H()
periph = PeripheralControllerFT232H()
sound = SoundControllerFT232H()
check_for_wrangler = Event()
print("test1")
wrangler = WranglerControllerFT232H(motor, check_for_wrangler)
print("test2")
#sound.play_target_spotted()

if __name__ == "__main__":
    print("test1")
    #check_joystick: Thread = Thread(target=wrangler.check_joystick_state(), name="check_joystick")
    #check_joystick.start()
    # WranglerRadioTest
    #x = wrangler.read_radio()
    print("-----------------")
    #rpint(x)

    motor.set_x_angle(0)
    motor.set_y_angle(0)
    time.sleep(2)
    sound.play_idle_beeping()
    #motor.idle(threadjoystickstate)

    #motor.rotate_x_relative(10)

    while True:
        x = wrangler.read_radio()
        is_firing, direction = x
        if is_firing:
            fire.fire()
        if direction == WranglerController.Direction.NOP:
            pass
        elif direction == WranglerController.Direction.UP:
            wrangler.increase_y_angle()

        elif direction == WranglerController.Direction.DOWN:
            wrangler.decrease_y_angle()

        elif direction == WranglerController.Direction.LEFT:
            wrangler.increase_x_angle()

        elif WranglerController.Direction.RIGHT:
            wrangler.decrease_x_angle()
            
