from __future__ import annotations
from src.RuntimeConfigs.depth_vision_configurable import DepthVisionConfigurable


class OakDStereoCamera(DepthVisionConfigurable):

    def get_depthvision(self) -> DepthVisionConfigurable:
        pass