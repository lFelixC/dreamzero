from copy import deepcopy

import cv2
import numpy as np

from droid.misc.time import time_ms

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None
    print("WARNING: You have not setup the RealSense cameras, and currently cannot use them")


def gather_realsense_cameras():
    all_realsense_cameras = []
    if rs is None:
        return all_realsense_cameras

    try:
        devices = rs.context().query_devices()
    except RuntimeError as exc:
        print(f"RealSense device query failed: {exc}")
        return all_realsense_cameras

    for device in devices:
        try:
            serial_number = device.get_info(rs.camera_info.serial_number)
            has_color_sensor = any(
                sensor.supports(rs.camera_info.name)
                and "rgb" in sensor.get_info(rs.camera_info.name).lower()
                for sensor in device.query_sensors()
            )
        except RuntimeError:
            continue

        if has_color_sensor:
            all_realsense_cameras.append(RealSenseCamera(serial_number))

    return all_realsense_cameras


resize_func_map = {"cv2": cv2.resize, None: None}


class RealSenseCamera:
    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self.is_hand_camera = False
        self.high_res_calibration = False
        self.current_mode = None
        self._pipeline = None
        self._profile = None
        self._intrinsics = {}
        self._warned_recording = False

        print("Opening RealSense: ", self.serial_number)

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
        camera_fps=30,
        **_,
    ):
        self.traj_image = image
        self.traj_resolution = tuple(resolution)
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        self.camera_fps = int(camera_fps)
        self.stream_resolution = (640, 480)
        self.resize_func = resize_func_map[resize_func]
        if self.resize_func is None and self.traj_resolution != (0, 0):
            self.resize_func = cv2.resize

    def set_calibration_mode(self):
        self.image = True
        self.skip_reading = False
        self.output_resolution = (0, 0)
        self._configure_camera()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.skip_reading = not self.image
        self.output_resolution = self.traj_resolution
        self._configure_camera()
        self.current_mode = "trajectory"

    def _configure_camera(self):
        self.disable_camera()
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        width, height = self.stream_resolution
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.camera_fps)

        try:
            self._profile = self._pipeline.start(config)
        except RuntimeError as exc:
            self._pipeline = None
            raise RuntimeError(f"RealSense camera {self.serial_number} failed to open: {exc}") from exc

        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(color_profile.get_intrinsics())
        }

    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        if not self._warned_recording:
            print(
                "Warning: RealSense recording is not implemented in this DROID wrapper; "
                f"skipping recording for {self.serial_number}."
            )
            self._warned_recording = True

    def stop_recording(self):
        return

    def _process_frame(self, frame):
        frame = np.asanyarray(frame.get_data()).copy()
        if self.output_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.output_resolution)

    def read_camera(self):
        if self.skip_reading:
            return {}, {}

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return {}, timestamp_dict

        color_frame = frames.get_color_frame()
        if not color_frame:
            return {}, timestamp_dict

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()
        received_time = color_frame.get_timestamp()
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time

        return {
            "image": {
                self.serial_number + "_left": self._process_frame(color_frame),
            }
        }, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
