from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.IO.stereo_camera import StereoCamera

from typing import TypedDict

STEREO_DIR: Path = Path("IOImplementations/TurretSRC/StereoCameras/StereoCalibration")


class Config(TypedDict):
    stereo_camera: StereoCamera
    left_stereo_path: Path
    right_stereo_path: Path
    num_photos_to_take: int
    exclusion_dir: Path
    chessboard_inner_pt_dim: tuple[int, int]
    square_size_mm: float
    skip_display: bool
    obj_pts_path: Path
    left_res_path: Path
    right_res_path: Path
    general_stereo_info_path: Path
    left_stereo_map_path: Path
    right_stereo_map_path: Path
    left_matcher_path: Path
    right_matcher_path: Path
    hyperparam_save_path: Path


config: Config = {
    # "stereo_camera": MakeshiftStereoCamera(Arducam(0), Arducam(1)),
    "left_stereo_path": STEREO_DIR / "stereo_images/calibration_images/stereo_l",
    "right_stereo_path": STEREO_DIR / "stereo_images/calibration_images/stereo_r",
    "num_photos_to_take": 30,
    "exclusion_dir": STEREO_DIR / "stereo_images/calibration_images",
    "chessboard_inner_pt_dim": (6, 9),
    "square_size_mm": 21,
    "skip_display": True,
    "obj_pts_path": STEREO_DIR / "saved_results/camera_calib/obj_pts.npz",
    "left_res_path": STEREO_DIR / "saved_results/camera_calib/left_cam_calib_results.npz",
    "right_res_path": STEREO_DIR / "saved_results/camera_calib/right_cam_calib_results.npz",
    "general_stereo_info_path": STEREO_DIR / "saved_results/camera_calib/general_stereo_info.npz",
    "left_stereo_map_path": STEREO_DIR / "saved_results/camera_calib/stereo_left.npz",
    "right_stereo_map_path": STEREO_DIR / "saved_results/camera_calib/stereo_right.npz",
    "left_matcher_path": Path("IOImplementations/TurretSRC/StereoCameras/StereoCalibration/stereo_images"
                              "/matcher_images/im2.png"),
    "right_matcher_path": Path("IOImplementations/TurretSRC/StereoCameras/StereoCalibration/stereo_images"
                               "/matcher_images/im6.png"),
    "hyperparam_save_path": STEREO_DIR / "saved_results/hyperparams/stereo_bm_default_values.json"
}
