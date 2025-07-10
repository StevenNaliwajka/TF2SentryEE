from __future__ import annotations

import threading

import numpy as np
import depthai as dai
from threading import Lock
from typing import cast, Optional
from TurretSRC.StereoCameras.oak_d_pipeline_component import OakDPipelineComponent
from graphlib import TopologicalSorter

from src.IO.stereo_camera import StereoCamera


class OakD(StereoCamera, OakDPipelineComponent):

    def __init__(self, dependency_graph: TopologicalSorter,
                 resolution: tuple[int, int] = (640, 480), cap_fps: int = 30,
                 use_structured_light: bool = False) -> None:

        # Nodes:
        self.left_cam: Optional[dai.node.Camera] = None
        self.right_cam: Optional[dai.node.Camera] = None
        self.stereo_depth: Optional[dai.node.StereoDepth] = None
        self.rgb_cam: Optional[dai.node.Camera] = None
        self.rgbd_node: Optional[dai.node.RGBD] = None

        # Outputs:
        self.left_out: Optional[dai.Node.Output] = None
        self.right_out: Optional[dai.Node.Output] = None
        self.rgb_out: Optional[dai.Node.Output] = None
        self.rgbd_out: Optional[dai.Node.Output] = None

        # MessageQueues
        self.left_messages: Optional[dai.MessageQueue] = None
        self.right_messages: Optional[dai.MessageQueue] = None
        self.rgb_d_messages: Optional[dai.MessageQueue] = None

        # Camera attributes
        self.resolution: tuple[int, int] = resolution
        self.capture_fps: int = cap_fps
        self.use_structured_light: bool = use_structured_light

        self._lock: threading.Lock = Lock()
        self.pipeline: dai.Pipeline = dai.Pipeline()
        self._device: dai.Device = self.pipeline.getDefaultDevice()

        # Print DeviceID, USB speed, and available cameras on the device
        print('DeviceID:', self._device.getDeviceInfo().getDeviceId())

        print('USB speed:', self._device.getUsbSpeed())

        print('Connected cameras:', self._device.getConnectedCameras())

        self.left_cam, self.right_cam = self._initialize_stereo_pair()
        self.rgb_cam = self._initialize_color_camera()

        self.dependencies: TopologicalSorter = dependency_graph
        self.handle_dependents()

        self.start_camera()

    def initialize_pipelines(self, parents: list[OakDPipelineComponent]) -> None:
        self.left_out = self.left_cam.requestOutput(self.resolution)
        self.right_out = self.right_cam.requestOutput(self.resolution)
        self.rgb_out = self.rgb_cam.requestOutput(self.resolution, type=dai.ImgFrame.Type.RGB888i)  # interleaved

        self.left_messages = self.left_out.createOutputQueue()
        self.right_messages = self.left_out.createOutputQueue()

        self.stereo_depth = self.pipeline.create(dai.node.StereoDepth)
        self.stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

        self.left_out.link(self.stereo_depth.left)
        self.right_out.link(self.stereo_depth.right)

        self.rgbd_node = self.pipeline.create(dai.node.RGBD).build()
        self.rgbd_node.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER)
        self.rgb_out.link(self.rgbd_node.inColor)
        self.stereo_depth.depth.link(self.rgbd_node.inDepth)
        self.rgb_d_messages = self.rgbd_node.rgbd.createOutputQueue()

        if self.use_structured_light:
            script: dai.node.Script = self.pipeline.create(dai.node.Script)
            # These values should be calibrated by you before you run this depending on your environment
            script.setScript("""
                            Device.setIrLaserDotProjectorIntensity(0.8)
                            Device.setIrFloodLightIntensity(0)
                            """)

    def get_images(self) -> tuple[np.ndarray, np.ndarray]:
        """
        This function returns the left and right stereo images used to make the disparity.
        Returns:
            A double containing the left and right stereo images (left,right).
            The number of channels depends on hardware. For the oak-d pro, it will be (h,w) greyscale.
        """
        left: dai.ADatatype = self.left_messages.get()
        right: dai.ADatatype = self.right_messages.get()
        if not isinstance(left, dai.ImgFrame):
            raise ValueError("Left message queue is not a imgframe")
        if not isinstance(right, dai.ImgFrame):
            raise ValueError("Right message queue is not a imgframe")
        left: dai.ImgFrame = cast(dai.ImgFrame, left)
        right: dai.ImgFrame = cast(dai.ImgFrame, right)

        return left.getCvFrame(), right.getCvFrame()

    def get_image_with_depth(self) -> np.ndarray:
        """
        This function will return an RGBD image.
        Returns:
            An RGB-D image of the format (height,width,channels).
            The channels will be in RGB order with the last channel being D (depth).
            The depth will be in the units specified as a hyperparameter above (millimeters by default)
        """
        image_with_depth: dai.ADatatype = self.rgb_d_messages.get()

        if not isinstance(image_with_depth, dai.RGBDData):
            raise RuntimeError("The rgbd queue returned back the wrong message type.")

        rgb_image: np.ndarray = image_with_depth.getRGBFrame().getCvFrame()
        depth_image: np.ndarray = image_with_depth.getDepthFrame().getCvFrame()
        return np.stack((rgb_image, depth_image), axis=-1)

    def release_interfaces(self) -> None:
        with self._lock:
            try:
                self.pipeline.stop()
                if not self._device.isClosed():
                    self._device.close()
            except RuntimeError as e:
                print("Error closing oak-d camera!", e)

    def get_intrinsic_matrices(self) -> np.ndarray:
        calibration: dai.CalibrationHandler = self._device.readCalibration()
        return np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.RGB, *self.resolution))

    def _initialize_color_camera(self) -> dai.node.Camera:
        """
        This function aims to initialize the color camera with the settings specified.
        """
        rgb_cam: dai.node.Camera = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.RGB
        )
        return rgb_cam

    def _initialize_stereo_pair(self) -> tuple[dai.node.Camera, dai.node.Camera]:
        """
        This function initializes the stereo pair for the oak-d camera.
        Returns:
            a double containing (left,right) camera pairs.
        """
        left_cam: dai.node.Camera = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.LEFT,
        )
        right_cam: dai.node.Camera = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.RIGHT,
        )

        return left_cam, right_cam

    def start_camera(self) -> None:
        """
        This function will start the oak-d camera.
        The reason it's a separate method is that you can still add additional pipeline steps afterward in
        detection or tracking or whatever else you want.
        """
        self.pipeline.start()

    def handle_dependents(self) -> None:
        """
        This function will handle the startup of all dependent nodes in order via DAG.
        The way that it will do this is by having all dependents declare ahead of time what they will need
        and this function will pass all parents to the dependents for them to do as they wish before
        passing along to their children which will do the same.
        This implementation is a bit of a nasty way to solve this problem, but still gives decent flexibility
        for not a crazy amount of complexity.
        Of course, most of the time, there will be maybe 3 dependents at most so this is still overkill.
        """
        parent_list: list[OakDPipelineComponent] = list(self.dependencies.static_order())
        curr_idx: int = 1
        while self.dependencies.is_active():
            node_group: tuple = self.dependencies.get_ready()
            curr_idx += len(node_group)
            for node in node_group:
                node: OakDPipelineComponent
                node.initialize_pipelines(parent_list[:curr_idx])
                self.dependencies.done(node)
