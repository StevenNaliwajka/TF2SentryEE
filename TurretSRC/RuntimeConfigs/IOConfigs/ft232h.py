from src.IO.io import IO
from src.RuntimeConfigs.io_configurable import IOConfigurable
from src.IOImplementations.TurretSRC.FT232HCode.motorcontrollerft232h import MotorControllerFT232H
from src.IOImplementations.TurretSRC.FT232HCode.soundcontrollerft232h import SoundControllerFT232H
from src.IOImplementations.TurretSRC.FT232HCode.firingcontrollerft232h import FiringControllerFT232H
from src.IOImplementations.TurretSRC.FT232HCode.peripheralcontrollerft232h import PeripheralControllerFT232H
from src.IOImplementations.TurretSRC.FT232HCode.wranglercontrollerft232h import WranglerControllerFT232H
from threading import Event


class FT232HIOBuilder(IOConfigurable):
    def get_io(self) -> IO:
        MOTOR_CONTROLLER: MotorControllerFT232H = MotorControllerFT232H()
        WRANGLER_EVENT: Event = Event()

        io: IO = IO(
            motor_controller=MOTOR_CONTROLLER,
            sound_controller=SoundControllerFT232H(),
            firing_controller=FiringControllerFT232H(),
            peripheral_controller=PeripheralControllerFT232H(),
            wrangler_controller=WranglerControllerFT232H(MOTOR_CONTROLLER, WRANGLER_EVENT),
            imu=None
        )

        return io
