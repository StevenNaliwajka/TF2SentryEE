# Created by Steven Naliwajka, https://github.com/ConvexBurrito5/TF2SentryEE
from src.IO.soundcontroller import SoundController
from playsound3 import playsound
import os

class SoundControllerFT232H(SoundController):

    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        parent_dir = os.path.dirname(current_dir)
        try:
            self.sentry_scan_path = os.path.join(parent_dir, "SoundFiles", "Sentry_scan.wav")
            self.sentry_spot_path = os.path.join(parent_dir, "SoundFiles", "Sentry_spot_client.wav")
            print("Initialized: Sound Controller")
        except:
            print("Sound Controller: Audio Path is incorrect")

    def play_idle_beeping(self):
        print(self.sentry_scan_path)
        playsound(self.sentry_scan_path)

    def play_target_spotted(self):
        #This plays "spotted" sound when called
        playsound(self.sentry_spot_path)
