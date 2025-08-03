from __future__ import annotations

import threading

import cv2
import numpy as np
import depthai as dai
from threading import Lock
from typing import cast

from numpy import typing as npt
from TurretSRC.RuntimeConfigs.DepthVision.oak_d_pipeline_component import OakDPipelineComponent

from src.IO.stereo_camera import StereoCamera


class OakD(StereoCamera, OakDPipelineComponent):

    def __init__(self) -> None:
        # We will use a hack to assign None initially while not allowing None later on.

        # Nodes:
        super().__init__()
        self.left_cam: dai.node.Camera = cast(dai.node.Camera, None)
        self.right_cam: dai.node.Camera = cast(dai.node.Camera, None)
        self.stereo_depth: dai.node.StereoDepth = cast(dai.node.StereoDepth, None)
        self.rgb_cam: dai.node.Camera = cast(dai.node.Camera, None)
        self.rgbd_node: dai.node.RGBD = cast(dai.node.RGBD, None)

        # Outputs:
        self.left_out: dai.Node.Output = cast(dai.Node.Output, None)
        self.right_out: dai.Node.Output = cast(dai.Node.Output, None)
        self.rgb_out: dai.Node.Output = cast(dai.Node.Output, None)
        self.rgbd_out: dai.Node.Output = cast(dai.Node.Output, None)

        # MessageQueues
        self.left_messages: dai.MessageQueue = cast(dai.MessageQueue, None)
        self.right_messages: dai.MessageQueue = cast(dai.MessageQueue, None)
        self.rgb_d_messages: dai.MessageQueue = cast(dai.MessageQueue, None)

        # Camera attributes
        self.resolution: tuple[int, int] = cast(tuple[int, int], None)
        self.capture_fps: int = cast(int, None)
        self.use_structured_light: bool = cast(bool, None)

        self._lock: threading.Lock = Lock()
        self.pipeline: dai.Pipeline = dai.Pipeline()
        self._device: dai.Device = self.pipeline.getDefaultDevice()

    def initialize_pipelines(self, parents: dict[str, OakDPipelineComponent]) -> None:

        self.left_cam, self.right_cam = self._initialize_stereo_pair()
        self.rgb_cam = self._initialize_color_camera()

        self.left_out = self.left_cam.requestOutput(self.resolution)
        self.right_out = self.right_cam.requestOutput(self.resolution)
        self.rgb_out = self.rgb_cam.requestOutput(self.resolution, dai.ImgFrame.Type.RGB888i)  # interleaved

        self.left_messages = self.left_out.createOutputQueue()
        self.right_messages = self.left_out.createOutputQueue()

        self.stereo_depth = self.pipeline.create(dai.node.StereoDepth)
        self.stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        self.stereo_depth.setRectifyEdgeFillColor(0)
        self.stereo_depth.enableDistortionCorrection(True)

        self.left_out.link(self.stereo_depth.left)
        self.right_out.link(self.stereo_depth.right)

        self.rgbd_node = self.pipeline.create(dai.node.RGBD).build()
        self.rgbd_node.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER)
        self.rgb_out.link(self.rgbd_node.inColor)
        self.stereo_depth.depth.link(self.rgbd_node.inDepth)
        self.rgb_out.link(self.stereo_depth.inputAlignTo)
        self.rgb_d_messages = self.rgbd_node.rgbd.createOutputQueue()

        if self.use_structured_light:
            script: dai.node.Script = self.pipeline.create(dai.node.Script)
            # These values should be calibrated by you before you run this depending on your environment
            script.setScript("""
                            Device.setIrLaserDotProjectorIntensity(0.8)
                            Device.setIrFloodLightIntensity(0)
                            """)

    def build_camera(self) -> None:
        """
        This will function as our real __init__()
        """
        # Print DeviceID, USB speed, and available cameras on the device
        print('DeviceID:', self._device.getDeviceInfo().getDeviceId())

        print('USB speed:', self._device.getUsbSpeed())

        print('Connected cameras:', self._device.getConnectedCameras())

        self.start_camera()
        pass

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

    def get_image_with_depth(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
        """
        This function will return an RGB image along with the corresponding depth.
        Returns:
            A double of the format:
            1. An RGB image of the format (height,width,channels).
            2. The depth map of each pixel in the unit specified earlier (default: millimeters)
        """
        image_with_depth: dai.ADatatype = self.rgb_d_messages.get()

        if not isinstance(image_with_depth, dai.RGBDData):
            raise RuntimeError("The rgbd queue returned back the wrong message type.")

        # getCVFrame automatically parses this as a BGR frame, so we have to flip the axes to get RGB.
        rgb_image: np.ndarray = image_with_depth.getRGBFrame().getCvFrame()[:, :, ::-1]
        depth_image: np.ndarray = image_with_depth.getDepthFrame().getCvFrame()
        return rgb_image, depth_image

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
            dai.CameraBoardSocket.RGB,
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

        while True:
            img, depth = self.get_image_with_depth()
            cv2.imshow("left", img)
            cv2.waitKey(1)
