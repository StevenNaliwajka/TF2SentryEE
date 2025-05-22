from __future__ import annotations
from typing import TYPE_CHECKING

import os
import numpy as np
import cv2
from pathlib import Path
import shutil
import re
from tqdm import tqdm
from dataclasses import dataclass

if TYPE_CHECKING:
    from IO.stereo_camera import StereoCamera

"""
These helper functions will help calibrate and set up the stereo camera pair.
This setup should only occur once per camera pair.
Note that these functions are not written in the most efficient way as to catch errors as fast as possible and 
raise exceptions as needed. 

Big thank you to the following resources for helping write some of these functions:
    Aryan Vij: https://github.com/aryanvij02/StereoVision
    Albert Armea: https://albertarmea.com/post/opencv-stereo-camera/
"""


def _assert_directories_same(left_stereo_path: Path, right_stereo_path: Path) -> list[str]:
    """
    This function will assert that the image names across two directories (the left and right stereo images)
    are the same and that there is no missing images.
    This function will raise ValueError if there is a discrepancy.
    params:
        left_stereo_path: Path the left stereo image directory
        right_stereo_path: Path the right stereo image directory
    returns:
        the list of image names shared across the two directories
    """
    left_files: list[str] = [file.name for file in left_stereo_path.iterdir()]
    right_files: list[str] = [file.name for file in right_stereo_path.iterdir()]
    left_files.sort()
    right_files.sort()
    if not left_files == right_files:
        raise ValueError("There is a disparity between the left directory and the right directory! Check your "
                         "left and right directory to make sure the number of files are the same and that "
                         "they are named the same.")
    return left_files


@dataclass
class CameraCalibrationResults:
    """
    This class will act as a struct containing information about the camera calibration
    params:
        image_points: the 2d points in the image plane. This refers to the checkerboard positions.
        camera_matrix: the matrix containing intrinsic values for the matrix such as focal length
            and optical centers. This matrix is after correction for the distortion values and thus,
            this matrix can be used later with initUndistortRectifyMap() and remap() to undistort your image.
        distortion_vect: the vector containing information about how the camera was distorted.
        rmse: Root Mean Squared Error: how accurate the camera calibration was. Anything under 1 is very good.
    """
    image_points: list[np.ndarray]
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_size: tuple
    rmse: float

    def get_calibration_info(self) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple, float]:
        """
        This helper function will make it easy to pass into stereo camera calibration by allowing for
        the use of the explode(unpack) operator * to do it in one line.
        return:
            a tuple containing (image_points, camera_matrix, distortion_coeffs, image_size) in that order.
        """
        return self.image_points, self.camera_matrix, self.distortion_coeffs, self.image_size, self.rmse


@dataclass
class StereoCalibrationResults:
    """
    This class will act as a struct containing information about the stereo pair.
    """
    essential_matrix: np.ndarray
    fund_mat: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    reproj_error: float

    def get_calibration_info(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        return self.essential_matrix, self.fund_mat, self.rotation_matrix, self.translation_vector, self.reproj_error


def take_photos(
        stereo_camera: StereoCamera,
        left_stereo_path: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_l/",
        right_stereo_path: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_r/",
        photos_to_take: int = 30,
        flush: bool = False
) -> None:
    """
    This function upon being called with start the stereo camera photo taking procedure.
    params:
    stereo_camera: A pointer to the stereo camera that you want to take the photos with.
    left_stereo_path: The directory that the left stereo images should be saved to.
    right_stereo_path: The directory that the right stereo images should be saved to.
    photos_to_take: how many photos should be taken. Default 30.
                    you probably don't want this to be a big number, otherwise you will take forever to calibrate.
    flush: if the left stereo and right stereo directories should be cleared before starting.
            If false, will just add more photos on top of the directory.
            Default: False.
    """
    WINDOW_NAME: str = "Images"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    photos_taken: int = 0

    if not (left_stereo_path.exists() or left_stereo_path.is_dir()):
        print(f"Left stereo directory not found, at {left_stereo_path}. Creating it now!")
        os.makedirs(left_stereo_path)

    if not (right_stereo_path.exists() or right_stereo_path.is_dir()):
        print(f"Left stereo directory not found, at {right_stereo_path}. Creating it now!")
        os.makedirs(right_stereo_path)

    last_number: int = 0
    if flush:
        user_response: str = input("We will DELETE the directory at " + str(left_stereo_path) + " and " +
                                   str(right_stereo_path) + "are you sure about this? (Y/N)")
        if user_response.upper() == 'Y':
            shutil.rmtree(left_stereo_path)
            shutil.rmtree(right_stereo_path)
            os.mkdir(left_stereo_path)
            os.mkdir(right_stereo_path)
        else:
            print("Cancelling deletion!!!")
            return
    else:

        file_names: list[str] = _assert_directories_same(left_stereo_path, right_stereo_path)
        if len(file_names) == 0:
            last_number = 0
        else:
            result: re.Match[str] = re.search(r'(\d+)', file_names[-1])
            # A lot going on here, but basically grab the numerically last file number
            # We cannot directly use sort because then 9 comes before 89 because of how sort works for strings.
            last_number = sorted(tuple(map(int, result.groups())))[-1]

    left_frame, right_frame = stereo_camera.get_images()
    top_text_location: int = left_frame.shape[0] * 80 // 100
    bottom_text_location: int = left_frame.shape[0] * 90 // 100

    while photos_taken < photos_to_take:
        left_frame, right_frame = stereo_camera.get_images()
        combined_image: np.ndarray = np.hstack((left_frame, right_frame))
        cv2.putText(combined_image, "Please press the spacebar to take a photo", (0, top_text_location),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(combined_image, "Photos left: " + str(photos_to_take - photos_taken), (0, bottom_text_location),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("Images", combined_image)
        user_input = cv2.waitKey(1)

        if user_input == ord(' '):
            photos_taken += 1
            print("Taking photo...")
            # This should be ok to sort later because numbers are lexicographically sortable through ascii
            # e.g. "0912" > "0091" and "0090" > "0089"
            # Note that the file extension here matters as cv2.imwrite() looks as that as the encoder.
            file_name: str = "image_" + str(last_number + photos_taken).zfill(4) + ".png"
            left_stereo_path / file_name
            cv2.imwrite(str(left_stereo_path / file_name), left_frame)
            print("Left image ", photos_taken, " saved.")
            cv2.imwrite(str(right_stereo_path / file_name), right_frame)
            print("Right image ", photos_taken, " saved.")
    cv2.destroyWindow(WINDOW_NAME)


def check_images(left_stereo_path: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_l",
                 right_stereo_path: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_r",
                 exclusion_dir: Path = Path(__file__).parent / "stereo_images/calibration_images/"
                 ) -> None:
    """
    This function will iterate through the images and gives the user the choice to keep them or to move them to
    the excluded folder (where you can delete them later if you wish)
    params:
        left_stereo_path: The directory where the left stereo images are stored.
        right_stereo_path: The directory where the right stereo images are stored.
        exclusion_dir: The parent directory where the left and right excluded images will be stored.
    """

    if not left_stereo_path.exists():
        raise ValueError("Did not find " + str(left_stereo_path) + ". Did you try running take_photos() "
                                                                   "with the right arguments?")
    if not right_stereo_path.exists():
        raise ValueError("Did not find " + str(right_stereo_path) + ". Did you try running take_photos() "
                                                                    "with the right arguments?")

    file_names: list[str] = _assert_directories_same(left_stereo_path, right_stereo_path)

    excluded_left: Path = exclusion_dir / "excluded_l"
    excluded_right: Path = exclusion_dir / "excluded_r"
    if not excluded_left.exists():
        print("Excluded left does not exist, creating it!")
        excluded_left.mkdir(parents=True)
        print("created a new directory at ", str(excluded_left))

    if not excluded_right.exists():
        print("Excluded right directory does not exist, creating it!")
        excluded_right.mkdir(parents=True)
        print("created a new directory at ", str(excluded_right))

    WINDOW_NAME: str = "Images"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    early_termination: bool = False

    for filename in file_names:
        if early_termination:
            break
        left_frame: cv2.typing.MatLike = cv2.imread(str(left_stereo_path / filename), cv2.IMREAD_UNCHANGED)
        right_frame: cv2.typing.MatLike = cv2.imread(str(right_stereo_path / filename), cv2.IMREAD_UNCHANGED)
        combined_image: np.ndarray = np.hstack((left_frame, right_frame))
        top_text_location: int = left_frame.shape[0] * 90 // 100
        cv2.putText(combined_image, "Press y to keep image, n to discard. q to quit.", (0, top_text_location),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, combined_image)
        # This is not mentioned in the docs anywhere but this method returns the last byte of the keycode in UTF.
        # This has been checked on 4.x release. This comment may not be true on future releases.
        # https://github.com/opencv/opencv/blob/master/modules/highgui/src/window.cpp

        while True:
            user_input = cv2.waitKey(0)
            if user_input == ord('y') or user_input == ord('Y'):
                break
            elif user_input == ord('n') or user_input == ord('N'):
                print("Moving files ", filename, "to the excluded directory")
                shutil.move(left_stereo_path / filename, excluded_left / filename)
                shutil.move(right_stereo_path / filename, excluded_right / filename)
                break
            elif user_input == ord('q') or user_input == ord('Q'):
                print("Terminating process early")
                early_termination = True
                break
            else:
                print("Unknown input, please try again")

    print("We are done with the image filter process.")

    cv2.destroyWindow(WINDOW_NAME)


def calibrate_both_cameras(
        chessboard_inner_pt_dim: tuple[int, int],
        square_size_mm: float,
        skip_display: bool = False,
        left_stereo_dir: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_l/",
        right_stereo_dir: Path = Path(__file__).parent / "stereo_images/calibration_images/stereo_r/",
        term_criteria: cv2.typing.TermCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        win_size: tuple[int, int] = (11, 11),
) -> tuple[list, CameraCalibrationResults, CameraCalibrationResults]:
    """
    This function aims to calibrate the stereo cameras. This will try to find the intrinsic and extrinsic
    distortion parameters that may warp the cameras.

    This follows closely from the official openCV docs:
    You really should read these if possible.
    https://docs.opencv.org/4.10.0/dc/dbb/tutorial_py_calibration.html
    as well as this tutorial from this website:
    https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/

    params:
    checkerboard_inner_pt_dim: This is actually the number of points BETWEEN checkerboard squares!
        This should be formatted in a (num_in_row,num_in_column) format.
        Please look at the openCV link above for a visual example.
        https://docs.opencv.org/4.10.0/calib_pattern.jpg In this example, the value would be (7,6)
    square_size_mm: This should be the size of the length of the square in mm.
    skip_display: This will control whether we should skip displaying the checkerboard pattern.
        This can be useful if you don't care to see the object points that were detected for each image.
    left_stereo_dir: the left stereo directory that we should look in for the left images
    right_stereo_dir: the right stereo directory that we should look in for the right images
    term_criteria: this triple will contain the termination_criteria, max_iterations, desired_acc in that order.
    win_size: the size of the window to use when trying to find points.
        A smaller value here means you can find the points more, but at the cost of computation speed.
        11x11 is a sane default. You should probably not change that, but if you could try 5x5 if you want to.
    returns:
        obj_pts: a list of object points in the real world space. We assume that the z axis is fixed (0).
        left_camera_info: a struct containing information about left camera calibration
        right_camera_info: a struct containing information about right camera calibration
    """

    # Prepare object points: such as (0,0,0), (1,0,0), ...
    objp: np.ndarray = np.zeros((1, chessboard_inner_pt_dim[0] * chessboard_inner_pt_dim[1], 3), np.float32)
    # The meshgrid basically denotes all the points of the checkerboard that we are looking for.
    # These points will be in between each of the squares. (You will see it later when you run this)
    # For simplicity, we will suppose that the board was kept stationary on some XY plane and that we only moved
    # the camera.
    # This means that Z = 0. And thus, we see in the below code, we don't set the z axis at all!
    objp[0, :, :2] = np.mgrid[0:chessboard_inner_pt_dim[0], 0:chessboard_inner_pt_dim[1]].T.reshape(-1, 2)

    # Furthermore, if we know the size, we can denote each point as the size in mm so instead of calling it
    # point (0,1), we can call it (0,21) (mm) etc.
    objp *= square_size_mm

    img_pts_l: list = []
    img_pts_r: list = []
    obj_pts: list = []

    dir_names: list[str] = _assert_directories_same(left_stereo_dir, right_stereo_dir)

    if len(dir_names) == 0:
        raise ValueError("No Entries inside your stereo directory!")

    # This will be set by the function.
    # This will be a height x width tuple.
    left_shape: tuple = ()
    right_shape: tuple = ()

    # This will take a while, so we are using a tqdm progress bar.
    print("This may take a couple of minutes (depending on how many image pairs you have), hang tight!")
    for file_name in tqdm(dir_names):
        # These will be the colored versions of the images for viewing pleasure later.
        original_left: cv2.typing.MatLike = cv2.imread(str(left_stereo_dir / file_name), cv2.IMREAD_UNCHANGED)
        original_right: cv2.typing.MatLike = cv2.imread(str(right_stereo_dir / file_name), cv2.IMREAD_UNCHANGED)

        # These will be in greyscale for the actual computation.
        left_grey: cv2.typing.MatLike = cv2.imread(str(left_stereo_dir / file_name), cv2.IMREAD_GRAYSCALE)
        right_grey: cv2.typing.MatLike = cv2.imread(str(right_stereo_dir / file_name), cv2.IMREAD_GRAYSCALE)

        left_shape = left_grey.shape
        right_shape = right_grey.shape

        if left_shape is not None and left_shape != left_grey.shape:
            raise ValueError("There is a disparity in the size of left image " + file_name)

        if right_shape is not None and right_shape != right_grey.shape:
            raise ValueError("There is a disparity in the size of right image " + file_name)

        left_shape = left_grey.shape
        right_shape = right_grey.shape

        has_corners_l, left_corners = cv2.findChessboardCorners(left_grey, chessboard_inner_pt_dim,
                                                                flags=cv2.CALIB_CB_FAST_CHECK)
        has_corners_r, right_corners = cv2.findChessboardCorners(right_grey, chessboard_inner_pt_dim,
                                                                 flags=cv2.CALIB_CB_FAST_CHECK)

        if not (has_corners_l and has_corners_r):
            if not has_corners_l:
                print("Unable to find chessboard corners for the left image with filename", file_name, "Skipping "
                                                                                                       "pair.")
            else:
                print("Unable to find chessboard corners for the right image with filename ", file_name, "Skipping "
                                                                                                         "pair.")

        obj_pts.append(objp)
        # Smaller window here will probably be better for finding the actual chessboards, but will take longer.
        # 11x11 (actual winsize (11*2+1 x 11*2+1) = 23x23) is good enough for most applications.
        # Change this to something like 5x5 if you have lots of compute power.
        cv2.cornerSubPix(left_grey, left_corners, win_size, (-1, -1), term_criteria)
        cv2.cornerSubPix(right_grey, right_corners, win_size, (-1, -1), term_criteria)
        cv2.drawChessboardCorners(original_left, chessboard_inner_pt_dim, left_corners, has_corners_l)
        cv2.drawChessboardCorners(original_right, chessboard_inner_pt_dim, right_corners, has_corners_r)
        if not skip_display:
            cv2.imshow('LeftCorners', original_left)
            cv2.imshow('RightCorners', original_right)
            print("Press any key to continue!")
            cv2.waitKey(0)

        img_pts_l.append(left_corners)
        img_pts_r.append(right_corners)

    # We invert the shape because for some reason the shape parameter
    # is in (width, height) format instead of the usual (height, width)
    inv_left_shape: tuple = left_shape[::-1]
    inv_right_shape: tuple = right_shape[::-1]

    # noinspection PyTypeChecker
    rmse_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(obj_pts, img_pts_l, inv_left_shape, None, None)
    new_mtx_l, _ = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, inv_left_shape, 1, inv_left_shape)

    # noinspection PyTypeChecker
    rmse_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(obj_pts, img_pts_r, inv_right_shape, None, None)
    new_mtx_r, _ = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, inv_right_shape, 1, inv_right_shape)

    left_camera_info: CameraCalibrationResults = \
        CameraCalibrationResults(img_pts_l, new_mtx_l, dist_l, inv_left_shape, rmse_l)

    right_camera_info: CameraCalibrationResults = \
        CameraCalibrationResults(img_pts_r, new_mtx_r, dist_r, inv_right_shape, rmse_r)

    if not skip_display:
        cv2.destroyWindow('LeftCorners')
        cv2.destroyWindow('RightCorners')

    return obj_pts, left_camera_info, right_camera_info


def calibrate_stereo(
        obj_pts: list,
        left_camera_info: CameraCalibrationResults,
        right_camera_info: CameraCalibrationResults,
        term_criteria: cv2.typing.TermCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
) -> tuple[StereoCalibrationResults, tuple, tuple]:
    img_pts_l, intrinsic_mat_l, dist_l, inv_left_shape, _ = left_camera_info.get_calibration_info()
    img_pts_r, new_mtx_r, dist_r, inv_right_shape, _ = right_camera_info.get_calibration_info()

    flags: int = 0
    flags |= cv2.CALIB_FIX_INTRINSIC  # Because this enum is a hex number (to allow for multiple calibration
    # options)
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental
    # matrix
    # noinspection PyTypeChecker
    reproj_error, intrinsic_mat_l, dist_l, new_mtx_r, dist_r, rot_mat, trans_vec, ess_matrix, fund_mat = \
        cv2.stereoCalibrate(
            obj_pts, img_pts_l,
            img_pts_r,
            intrinsic_mat_l, dist_l,
            new_mtx_r,
            dist_r,
            inv_left_shape,
            term_criteria, flags)

    general_stereo_info = StereoCalibrationResults(ess_matrix, fund_mat, rot_mat, trans_vec, reproj_error)

    print("Reprojection error for the stereo pair is ", reproj_error)

    tolerance: float = 3.0
    if not np.all(np.isclose(np.identity(3), rot_mat, atol=tolerance)):
        raise RuntimeError("Your stereo camera has more than " + str(tolerance) + " degrees of rotation! "
                                                                                  "Are you sure that your cameras "
                                                                                  "are parallel?")
    baseline_length_mm: float = np.linalg.norm(trans_vec)

    rectify_scale = 1

    # we want to determine the rotation and vertical offset of this stereo pair in order to rectify the image pair.

    # noinspection PyTypeChecker
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roi_r = cv2.stereoRectify(intrinsic_mat_l, dist_l, new_mtx_r,
                                                                               dist_r,
                                                                               inv_left_shape, rot_mat, trans_vec,
                                                                               rectify_scale, (0, 0))

    # We use CV_16SC2 here because we want to compute the stereo disparity map.
    left_stereo_map: tuple[np.ndarray, np.ndarray] = cv2.initUndistortRectifyMap(intrinsic_mat_l, dist_l, rect_l,
                                                                                 proj_mat_l,
                                                                                 inv_left_shape, cv2.CV_16SC2)
    # Of format (uint16, uint16)
    right_stereo_map: tuple[np.ndarray, np.ndarray] = cv2.initUndistortRectifyMap(new_mtx_r, dist_r, rect_r, proj_mat_r,
                                                                                  inv_right_shape, cv2.CV_16SC2)
    print(baseline_length_mm)

    return general_stereo_info, left_stereo_map, right_stereo_map


def save_all_results(
        obj_pts: list,
        left_camera_info: CameraCalibrationResults,
        right_camera_info: CameraCalibrationResults,
        general_stereo_info: StereoCalibrationResults,
        left_stereo_map: tuple,
        right_stereo_map: tuple,
        obj_pts_path: Path = Path(__file__).parent / "saved_results/camera_calib/obj_pts.npz",
        left_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/left_cam_calib_results.npz",
        right_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/right_cam_calib_results.npz",
        general_stereo_info_path: Path = Path(__file__).parent / "saved_results/camera_calib/general_stereo_info.npz",
        left_stereo_map_path: Path = Path(__file__).parent / "saved_results/camera_calib/left_stereo_map.npz",
        right_stereo_map_path: Path = Path(__file__).parent / "saved_results/camera_calib/right_stereo_map.npz"
) -> None:
    """
    This beefy function will take all the parameters that you currently have as a python object and write them to
    disk as npz. (numpy zipped)
    params:
        obj_pts: the list of object points used to calibrate the cameras
        left_camera_info: the left camera struct that contains information on the left camera
        right_camera_info: the right camera struct that contains information on the right camera
        general_stereo_info: general information about the stereo pair
        left_stereo_map: a double containing (map1,map2) for the left stereo rectification
        right_stereo_map: a double containing (map1,map2) for the right stereo rectification
        everything else is just the path of where to save it to.
    """
    if not isinstance(obj_pts, list):
        raise TypeError("obj_pts must be a list.")

    if not (isinstance(left_stereo_map, tuple) and len(left_stereo_map) == 2):
        raise ValueError("left_stereo_map must be a tuple with two elements.")

    if not (isinstance(right_stereo_map, tuple) and len(right_stereo_map) == 2):
        raise ValueError("right_stereo_map must be a tuple with two elements.")

    for path in [obj_pts_path, left_res_path, right_res_path, general_stereo_info_path,
                 left_stereo_map_path, right_stereo_map_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(obj_pts_path, obj_pts=obj_pts)
    print("Obj points have been successfully saved to ", str(obj_pts_path))

    img_pts_l, new_mtx_l, dist_l, inv_left_shape, rmse_l = left_camera_info.get_calibration_info()
    img_pts_r, new_mtx_r, dist_r, inv_right_shape, rmse_r = right_camera_info.get_calibration_info()

    np.savez_compressed(left_res_path,
                        img_pts_l=np.stack(img_pts_l),  # Convert the list of points into a nxk matrix.
                        new_mtx_l=new_mtx_l,
                        dist_l=dist_l,
                        inv_left_shape=inv_left_shape,
                        rmse_l=rmse_l,
                        map1=left_stereo_map[0],
                        map2=left_stereo_map[1],
                        )

    print("Left camera calibration parameters have been successfully saved to ", str(left_res_path))

    np.savez_compressed(right_res_path,
                        img_pts_r=np.stack(img_pts_r),
                        new_mtx_r=new_mtx_r,
                        dist_r=dist_r,
                        inv_right_shape=inv_right_shape,
                        rmse_r=rmse_r,
                        map1=right_stereo_map[0],
                        map2=right_stereo_map[1]
                        )

    print("Right camera calibration parameters have been successfully saved to ", str(right_res_path))

    np.savez_compressed(general_stereo_info_path,
                        ess_mat=general_stereo_info.essential_matrix,
                        fund_mat=general_stereo_info.fund_mat,
                        rot_mat=general_stereo_info.rotation_matrix,
                        trans_vect=general_stereo_info.translation_vector,
                        reproj_error=general_stereo_info.reproj_error
                        )
    print("General stereo information has been successfully saved to ", str(general_stereo_info_path))

    np.savez_compressed(left_stereo_map_path,
                        map1=left_stereo_map[0],
                        map2=left_stereo_map[1]
                        )
    print("Left stereo map path has been successfully saved to ", str(left_stereo_map_path))

    np.savez_compressed(right_stereo_map_path,
                        map1=right_stereo_map[0],
                        map2=right_stereo_map[1]
                        )
    print("Right stereo map path has been successfully saved to ", str(right_stereo_map_path))


def load_requested_results(
        requested_results: set[str] = None,
        obj_pts_path: Path = Path(__file__).parent / "saved_results/camera_calib/obj_pts.npz",
        left_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/left_cam_calib_results.npz",
        right_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/right_cam_calib_results.npz",
        general_stereo_info_path: Path = Path(__file__).parent / "saved_results/camera_calib/general_stereo_info.npz",
        left_stereo_map_path: Path = Path(__file__).parent / "saved_results/camera_calib/left_stereo_map.npz",
        right_stereo_map_path: Path = Path(__file__).parent / "saved_results/camera_calib/right_stereo_map.npz"
) -> dict:
    """
    This function will load all requested results specified or all of them if not specified.
    You better make sure that you have a path defined for the results you request, otherwise the default
    will be used.
    params:
        requested_results: The results you want to request. This should be a set that contain strings from the 
            following: "obj_pts", "left_camera_info", "right_camera_info", "general_stereo_info_path",
            "left_stereo_map_path","right_stereo_map_path".
            If you include the string inside the set, you must also specify the respective path, or you will be given
            the default path.
            You may also choose to omit any strings you don't need results for.
    Returns:
        a dictionary of the requested items from requested_results or all if not specified.
        The keys will be the exact same as the strings used for requested_results.
        obj_pts: the list of object points
        left_camera_info: the left camera info defined in the struct `CameraCalibrationResults`
        right_camera_info: the right camera info defined in the struct `CameraCalibrationResults`
        general_stereo_info: the general stereo pair info defined in the struct `StereoCalibrationResults`
        left_stereo_map: a double containing the map to rectify the left stereo map. Of the form (map1,map2)
        right_stereo_map: a double containing the map to rectify the right stereo map. Of the form (map1,map2)
    """

    if requested_results is None:
        requested_results = {
            "obj_pts",
            "left_camera_info",
            "right_camera_info",
            "general_stereo_info_path",
            "left_stereo_map_path",
            "right_stereo_map_path"
        }

    results: dict = {}

    if "obj_pts" in requested_results:
        with np.load(obj_pts_path) as data:
            results["obj_pts"] = list(data["obj_pts"])

    if "left_camera_info" in requested_results:
        with np.load(left_res_path) as data:
            results["left_camera_info"] = CameraCalibrationResults(
                data["img_pts_l"], data["intrinsic_matrix_l"],
                data["dist_l"], data["inv_left_shape"], data["rmse_l"]
            )

    if "right_camera_info" in requested_results:
        with np.load(right_res_path) as data:
            results["right_camera_info"] = CameraCalibrationResults(
                data["img_pts_r"], data["instrinsic_matrix_r"],
                data["dist_r"], data["inv_right_shape"], data["rmse_r"]
            )

    if "general_stereo_info" in requested_results:
        with np.load(general_stereo_info_path) as data:
            results["general_stereo_info"] = StereoCalibrationResults(
                essential_matrix=data["ess_mat"],
                fund_mat=data["fund_mat"],
                rotation_matrix=data["rot_mat"],
                translation_vector=data["trans_vect"],
                reproj_error=data["reproj_error"]
            )

    if "left_stereo_map_path" in requested_results:
        with np.load(left_stereo_map_path) as data:
            results["left_stereo_map"] = (data["map1"], data["map2"])

    if "right_stereo_map_path" in requested_results:
        with np.load(right_stereo_map_path) as data:
            results["right_stereo_map"] = (data["map1"], data["map2"])
    return results 


def get_focal_lengths_px(
        left_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/left_cam_calib_results.npz",
        right_res_path: Path = Path(__file__).parent / "saved_results/camera_calib/right_cam_calib_results.npz",
) -> tuple:
    """
    params:
        obj_pts_path: the name of the object points file that we are going to try to read from.
            Essentially the object points that you used to calibrate the left and right camera.
        left_res_path: same as obj_pts_filename but instead for the left camera results dataclass.
            Note that we will not actually store the dataclass python object, but each element seperately.
        right_res_path: same as left_res_filename but for the right camera results.
    return:
        This function will return a double of the focal lengths (fx,fy) for left camera and right camera.
        the format will be ((left_fx,left_fy),(right_fx,right_fy))
    """
    results: dict = load_requested_results({"left_camera_info", "right_camera_info"},
                                           left_res_path=left_res_path,
                                           right_res_path=right_res_path)
    
    left_cam_info: CameraCalibrationResults = results["left_camera_info"]
    right_cam_info: CameraCalibrationResults = results["right_camera_info"]

    return (
        (left_cam_info.camera_matrix[0][0], left_cam_info.camera_matrix[1][1]),
        (right_cam_info.camera_matrix[0][0], right_cam_info.camera_matrix[1][1])
    )
