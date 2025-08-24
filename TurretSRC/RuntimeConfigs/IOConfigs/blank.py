from src.IO.io import IO
from src.RuntimeConfigs.io_configurable import IOConfigurable


class BlankIO(IOConfigurable):
    def get_io(self) -> IO:
        """
        This is legal because we will replace the None here with the proper Blank implementations later in the
        sentry context class.
        """
        return IO(
            motor_controller=None,
            sound_controller=None,
            firing_controller=None,
            peripheral_controller=None,
            wrangler_controller=None,
            imu_controller=None
        )
