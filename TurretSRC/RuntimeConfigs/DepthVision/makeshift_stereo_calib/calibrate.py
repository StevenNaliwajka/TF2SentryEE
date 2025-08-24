from __future__ import annotations
import argparse
import TurretSRC.StereoCameras.StereoCalibration.stereo_calibrate as calibrator
from src import IOImplementations as calib_config
from TurretSRC.StereoCameras.StereoCalibration.stereo_bm_calibrator import StereoBMCalibrator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TurretSRC.StereoCameras.StereoCalibration.stereo_matcher_calibrator import \
        StereoMatcherCalibrator
    from TurretSRC.RuntimeConfigs.DepthVision.makeshift_stereo_calib.calibration_config import Config

if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This python script will help you get your stereo camera quickly set up"
    )

    parser.add_argument("-f", "--flush", action="store_true",
                        help="Flushes the stereo images and allows you to flush the images already taken"
                             "Flushing will DELETE the previously stored images")
    args = parser.parse_args()
    config: Config = calib_config.config

    calibrator.take_photos(config["stereo_camera"],
                           config["left_stereo_path"],
                           config["right_stereo_path"],
                           config["num_photos_to_take"],
                           args.flush)

    calibrator.check_images(config["left_stereo_path"], config["right_stereo_path"], config["exclusion_dir"])

    obj_pts, left_camera_info, right_camera_info = calibrator.calibrate_both_cameras(
        config["chessboard_inner_pt_dim"], config["square_size_mm"],
        config["skip_display"],
        config["left_stereo_path"], config["right_stereo_path"])

    general_stereo_info, left_stereo_map, right_stereo_map = calibrator.calibrate_stereo(obj_pts, left_camera_info,
                                                                                         right_camera_info)

    calibrator.save_all_results(obj_pts,
                                left_camera_info,
                                right_camera_info,
                                general_stereo_info,
                                left_stereo_map,
                                right_stereo_map,
                                config["obj_pts_path"],
                                config["left_res_path"],
                                config["right_res_path"],
                                config["general_stereo_info_path"],
                                config["left_stereo_map_path"],
                                config["right_stereo_map_path"])

    stereo_bm_calib: StereoMatcherCalibrator = StereoBMCalibrator(config["left_matcher_path"],
                                                                  config["right_matcher_path"],
                                                                  config["left_stereo_map_path"],
                                                                  config["right_stereo_map_path"],
                                                                  config["hyperparam_save_path"])

    stereo_bm_calib.tune_disparity_params()

    print("Calibration successful! Load your parameters later from the files that you just saved.")
