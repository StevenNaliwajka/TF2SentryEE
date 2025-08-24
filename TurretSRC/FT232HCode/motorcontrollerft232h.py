# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
import time
import digitalio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from threading import Event
from src.IO.motorcontroller import MotorController
from TurretSRC.FT232HCode.singletonboardft232h import SingletonBoardFT232H


class MotorControllerFT232H(MotorController):

    def __init__(self):
        # IF YOU HAVE DIFFERENT MOTORS THAN MINE, UPDATE THESE VALUES BEFORE RUNNING
        # ------------------------
        self.MAX_X_ANGLE = 270
        self.MAX_Y_ANGLE = 180
        self.MIN_X_ANGLE = 0
        self.MIN_Y_ANGLE = 0
        self.X_MIN_PULSE = 500
        self.X_MAX_PULSE = 2500
        self.Y_MIN_PULSE = 500
        self.Y_MAX_PULSE = 2500
        # ------------------------
        self.total_fails = 0

        # I2C register data
        self.x_position = 0
        self.y_position = 0
        self.fire_state = 0
        self.move_state = 0
        self.wrangler_status = 0

        # print(SCL)
        # print(SDA)
        # create i2c Bus
        self.i2c_bus = SingletonBoardFT232H().i2c_bus

        try:
            self._SETUP_PIN = digitalio.DigitalInOut(board.C2)
            self._SETUP_PIN.direction = digitalio.Direction.OUTPUT
            # Create the PCA9685 slave on the I2C bus
            self.PCA = PCA9685(self.i2c_bus)
            # Setup the SCL Freq
            self.PCA.frequency = 50
            self.PCA.channels[0].duty_cycle = 0x7FFF
            # DEFINE the servos HERE. In this case there is only one PCA9685 board here.
            # 500 & 2500 are the magic numbers for the Servos. Ripped off amazon page
            self.xAxis = servo.Servo(pwm_out=self.PCA.channels[0], min_pulse=self.X_MIN_PULSE,
                                     max_pulse=self.X_MAX_PULSE, actuation_range=self.MAX_X_ANGLE)
            self.yAxis = servo.Servo(pwm_out=self.PCA.channels[1], min_pulse=self.Y_MIN_PULSE,
                                     max_pulse=self.Y_MAX_PULSE, actuation_range=self.MAX_Y_ANGLE)
            print("Initialized: MotorController")

        except:
            print("MotorController: Failure to Init, PCA9685 chip not connected!")
            self.i2c_bus.unlock()

        try:
            print("MotorController: Beginning motor calibration.")
            self._calibrate_servo_potentiometers()
        except:
            print("MotorController: Motor calibration failed.")
        else:
            print("MotorController: Motor calibration successful.")

    """
    Returns: the angle that the XY servo is currently in.
    """

    def get_x_angle(self) -> int:
        self._update_position()
        return round(self.x_position, 12)

    """
    Returns: the angle that the YZ servo is currently in.
    """

    def get_y_angle(self) -> float:
        self._update_position()
        # print("MotorCtrl: Current Position is ")
        # print(self.y_position)
        return round(self.y_position, 12)

    """
    Pass 0-MAX_X_ANGLE in, Sends CMD for XY(X) servo motor to turn.
    If the angle provided is too big, it will set the current_angle to MAX_Y_ANGLE
        and return False.
    If the angle is too small, it will set the current_angle to 0 
        and return False.
    angle: float the raw(not relative) angle that it should turn to.
    Returns True if 0<=angle<=MAX_X_ANGLE
        False if this condition is not met.
    """

    def set_x_angle(self, angle: float) -> bool:
        # Sets the current angle. MAX is MAX_X_ANGLE DEG
        if angle < self.MIN_X_ANGLE:
            self.xAxis.angle = self.MIN_X_ANGLE
            return False
        elif angle > self.MAX_X_ANGLE:
            self.xAxis.angle = self.MAX_X_ANGLE
            return False
        else:
            self.xAxis.angle = angle
            return True

    """
    Pass 0-MAX_Y_ANGLE in, Sends CMD for YZ (Y) motor to turn.
    If the angle provided is too big, it will set the current_angle to MAX_Y_ANGLE
        and return False.
    If the angle is too small, it will set the current_angle to 0 
        and return False.
    angle: float the raw(not relative) angle that it should turn to.
    Returns True if 0<=angle<=MAX_Y_ANGLE
        False if this condition is not met.
    """

    def set_y_angle(self, angle: float) -> bool:
        # Sets the current angle. Max is MAX_Y_ANGLE DEG
        if angle < self.MIN_Y_ANGLE:
            self.yAxis.angle = self.MIN_Y_ANGLE
            return False
        elif angle > self.MAX_Y_ANGLE:
            self.yAxis.angle = self.MAX_Y_ANGLE
            return False
        else:
            self.yAxis.angle = angle
            # self.yAxis.angle = 180
            # self.yAxis.angle += 3
            return True

    """
    This function will rotate the XY (X) motor by a relative input.
    For example, if you pass in 10 degrees and the motor is currently
    at 100 degrees, then the servo will spin to 110 degrees.
    It also accepts negative inputs, so -10 is also valid.

    *If the angle you specified was too much, it will just move to the max
    in that direction*
    E.g. If you put set angle = 1000, then it will just rotate to MAX_X_ANGLE.
    Same applies for the negative direction, if you try to spin too far, it
    will just go to zero.
    
    angle: float how many degrees we want to increase/decrease the servo
    Returns: True if 0 <= (angle + current_angle) <= MAX_X_ANGLE
        False if this condition is not met.
        Beware of floating point errors as you could get something that
        should be true, but isnt true.
    """

    def rotate_x_relative(self, angle: float) -> bool:
        xCurrent: float = self.get_x_angle()
        print(xCurrent)
        xNew = angle + xCurrent
        print(xNew)
        if xNew < self.MIN_X_ANGLE:
            self.xAxis.angle = self.MIN_X_ANGLE
            return False
        elif xNew > self.MAX_X_ANGLE:
            self.xAxis.angle = self.MAX_X_ANGLE
            return False
        else:
            self.set_x_angle(xNew)
            return True

    """
    This function will rotate the YZ (Y) motor by a relative input.
    For example, if you pass in 10 degrees and the motor is currently
    at 100 degrees, then the servo will spin to 110 degrees.
    It also accepts negative inputs, so -10 is also valid.

    *If the angle you specified was too much, it will just move to the max
    in that direction*
    E.g. If you put set angle = 1000, then it will just rotate to MAX_Y_ANGLE.
    Same applies for the negative direction, if you try to spin too far, it
    will just go to zero.
    
    angle: float how many degrees we want to increase/decrease the servo
    Returns: True if 0 <= (angle + current_angle) <= MAX_Y_ANGLE
        False if this condition is not met.
        Beware of floating point errors as you could get something that
        should be true, but isnt true.
    """

    def rotate_y_relative(self, angle: float) -> bool:
        yCurrent: float = self.get_y_angle()
        yNew = angle + yCurrent
        if yNew < self.MIN_Y_ANGLE:
            self.yAxis.angle = self.MIN_Y_ANGLE
            return False
        elif yNew > self.MAX_Y_ANGLE:
            self.yAxis.angle = self.MAX_Y_ANGLE
            return False
        else:
            self.set_y_angle(yNew)
            return True

    # set#Angle wrappers.
    def set_x_min(self) -> None:
        self.set_x_angle(self.MIN_X_ANGLE)

    def set_y_min(self) -> None:
        self.set_y_angle(self.MIN_Y_ANGLE)

    def set_x_max(self) -> None:
        self.set_x_angle(self.MAX_X_ANGLE)

    def set_y_max(self) -> None:
        self.set_y_angle(self.MAX_Y_ANGLE)

    def pause_movement(self):
        self.set_x_angle(self.get_x_angle())
        self.set_y_angle(self.get_y_angle())

    def pause_x(self):
        x = self.get_x_angle()
        self.set_x_angle(x)

    def pause_y(self):
        x = self.get_y_angle()
        self.set_y_angle(x)

    def idle(self, stopCon: Event) -> None:
        # There is no way to make PWM signals convey speed.
        # To get an IDLE state for the turret we just move a few degrees
        # at a time and wait a few ms between.
        # Runs till event is set.

        # IDLE NEEDS TO BE UPDATED TO BE GENERAL
        while True:
            # print ("IN IDLE")
            for _ in range(50):
                # print("rotate x")
                if stopCon.isSet():
                    stopCon.clear()
                    return
                self.rotate_x_relative(5.4)
                time.sleep(.3)

            for _ in range(50):
                if stopCon.isSet():
                    stopCon.clear()
                    return
                self.rotate_x_relative(-5.4)
                time.sleep(.3)

    def _update_position(self, failCounter=1, limit=10) -> None:
        if failCounter >= limit:
            raise RuntimeError("MotorController: Got bad data " + limit + " times in a row. "
                                                                          "Please check the hardware.")
        # Updates X and Y position from Using I2C comm with the aurduino that is monitoring the potentiometers
        # Check if I2C is open, if so take ctrl
        while not self.i2c_bus.try_lock():
            print("Motor Controller: I2C Bus is locked, Unable to update position")
            pass
        try:
            # print("Motor Controller Taken control of I2C")
            result = bytearray(14)
            # Address of aurduino is 0x55, load data into it
            self.i2c_bus.readfrom_into(0x55, result)
            # Convert bits into ints and load into array
            integers = [int(byte) for byte in result]
            # print(integers)
            if integers[0] + integers[2] == (self.MAX_X_ANGLE + 1):
                if integers[2] == 0:
                    integers[0] -= 1
                else:
                    integers[2] -= 1
            if integers[4] + integers[6] == (self.MAX_Y_ANGLE + 1):
                if integers[6] == 0:
                    integers[4] -= 1
                else:
                    integers[6] -= 1
            # error checking of data, integers[7] is designed to be empty. IF bad data its typical 255
            if integers[7] != 0:
                # if bad data. print
                print('MotorController: InvalidData: #{}'.format(failCounter))
                print(integers)
                # time.sleep(.2)
                # give bus back and recursive call until good data pulled
                self.i2c_bus.unlock()
                self.total_fails += 1
                self._update_position(failCounter=failCounter + 1)
            else:
                # update position status
                # Reset the fail counter
                failCounter = 0
                # self._raw_x_pos = integers[0] + integers[2]
                # self._raw_y_pos = integers[4]
                self.x_position = integers[0] + integers[2]
                self.y_position = integers[4] + integers[6]
                # print("Current Y position:")
                # print(self.y_position)
                self.fire_state = integers[8]
                self.move_state = integers[10]
                self.wrangler_status = integers[12]
        except:
            # Error catching. Happens when the aurduino cant be pinged from the I2C bus
            print("MotorController: Aurduino not connected. I2C returning Bad Data")
            self.i2c_bus.unlock()
        finally:
            # Finish by giving up the bus
            self.i2c_bus.unlock()

    def _calibrate_servo_potentiometers(self) -> None:
        self._SETUP_PIN = True
        print("MotorController: Turning X Servo to %s DEG." % self.MIN_X_ANGLE)
        print("MotorController: Turning Y Servo to %s DEG." % self.MIN_Y_ANGLE)
        self.set_x_min()
        self.set_y_min()
        time.sleep(1.5)
        print("MotorController: Turning X Servo to %s DEG." % self.MAX_X_ANGLE)
        print("MotorController: Turning Y Servo to %s DEG." % self.MAX_Y_ANGLE)
        self.set_x_max()
        self.set_y_max()
        time.sleep(1.5)
        self._SETUP_PIN = False
        time.sleep(.3)
