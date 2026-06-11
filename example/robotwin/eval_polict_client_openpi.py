import sys
import os
import subprocess
import time
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ModuleNotFoundError:
    plt = None
    FigureCanvas = None
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
robowin_root = REPO_ROOT / "third_party" / "RoboTwin"
for path in (REPO_ROOT, robowin_root, robowin_root / "script"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import os
os.chdir(robowin_root)

CONFIGS_PATH = None
UnStableError = None

import numpy as np
from pathlib import Path
from collections import deque
import traceback

try:
    import yaml
except ModuleNotFoundError:
    yaml = None
from datetime import datetime
import importlib
import argparse
import pdb
from example.robotwin.geometry import euler2quat
import numpy as np

generate_episode_descriptions = None
import traceback

import imageio
import numpy as np
from pathlib import Path
try:
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError:
    R = None
import json
from pathlib import Path

SAVE_COMPARISON_VIDEO = os.getenv("SAVE_COMPARISON_VIDEO", "1").lower() not in ("0", "false", "no")
EPISODE_TIMEOUT_SEC = float(os.getenv("EPISODE_TIMEOUT_SEC", "900"))


class EpisodeTimeoutError(RuntimeError):
    pass


def require_yaml():
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to run RoboTwin eval. Activate the RoboTwin eval environment or install pyyaml.")
    return yaml


def import_policy_client():
    try:
        from eval_utils.policy_client import WebsocketClientPolicy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "websockets is required to connect to the DreamZero policy server. "
            "Activate the DreamZero eval environment or install websockets."
        ) from exc
    return WebsocketClientPolicy


def import_robotwin_runtime():
    global CONFIGS_PATH, UnStableError, generate_episode_descriptions
    from envs import CONFIGS_PATH as robotwin_configs_path
    from envs.utils.create_actor import UnStableError as robotwin_unstable_error
    from description.utils.generate_episode_instructions import generate_episode_descriptions as robotwin_generate_episode_descriptions
    from example.robotwin.test_render import Sapien_TEST

    CONFIGS_PATH = robotwin_configs_path
    UnStableError = robotwin_unstable_error
    generate_episode_descriptions = robotwin_generate_episode_descriptions
    return Sapien_TEST

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def append_jsonl(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def resize_image(img, size):
    if cv2 is not None:
        return cv2.resize(img, size)
    target_w, target_h = size
    y_idx = np.linspace(0, img.shape[0] - 1, target_h).astype(np.int64)
    x_idx = np.linspace(0, img.shape[1] - 1, target_w).astype(np.int64)
    return img[y_idx][:, x_idx]


def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    h, w, _ = img.shape
    bar_height = 40
    
    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    if cv2 is None:
        return np.vstack([title_bar, img])
    
    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5
    
    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return np.vstack([title_bar, img])

def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    quat: [rx, ry, rz, rw] format
    Return: [roll, pitch, yaw] (radians)
    """
    quat = np.asarray(quat, dtype=np.float64)
    if R is None:
        x, y, z, w = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.sign(sinp) * np.pi / 2 if abs(sinp) >= 1 else np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    # scipy uses [x, y, z, w] format
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
    return euler


def quat_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)

def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    
    Input data format: [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_rw, left_gripper,
                   right_x, right_y, right_z, right_rx, right_ry, right_rz, right_rw, right_gripper]
    Total 16 dimensions
    """
    if plt is None or FigureCanvas is None:
        img = np.zeros((800, 1400, 3), dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(img, "matplotlib not installed", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(img, f"step {step_idx}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        return img

    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)
    
    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1
    
    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]
    
    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)
    
    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []
        
        for action in history_subset:
            # Left arm quaternion to Euler angles
            left_quat = action[3:7]  # [rx, ry, rz, rw]
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)
            
            # Right arm quaternion to Euler angles
            right_quat = action[11:15]  # [rx, ry, rz, rw]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)
        
        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)
        
        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label='left_x', color='r', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label='left_y', color='g', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label='left_z', color='b', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label='left_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Position (m)')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label='left_roll', color='c', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label='left_pitch', color='m', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label='left_yaw', color='y', linewidth=1.5)
        ax2.set_ylabel('Rotation (rad)')
        ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label='right_x', color='r', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 9], label='right_y', color='g', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 10], label='right_z', color='b', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 15], label='right_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Position (m)')
        ax3.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label='right_roll', color='c', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 1], label='right_pitch', color='m', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 2], label='right_yaw', color='y', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel('Step')
    ax4.set_xlabel('Step')
    
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]
    
    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        
    plt.close(fig)
    return img


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video) 
    else:
        n_imagined = 0
    n_frames = n_real # Based on real observation frames
    
    print(f"Saving video: Real {n_real} frames, Imagined {n_imagined} frames...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs.get("observation.images.cam_left", obs.get("observation.images.cam_left_wrist"))
        cam_right = obs.get("observation.images.cam_right", obs.get("observation.images.cam_right_wrist"))
        if cam_left is None or cam_right is None:
            raise KeyError("Missing left/right camera images in observation payload")

        base_h = cam_high.shape[0]
        
        def resize_h(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = resize_image(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        row_real = np.hstack([
            resize_h(cam_high, base_h), 
            resize_h(cam_left, base_h), 
            resize_h(cam_right, base_h)
        ])
        
        row_real = np.ascontiguousarray(row_real)

        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        if imagined_video is not None and i < n_imagined:
            img_frame = imagined_video[i]
            if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)

            h = int(img_frame.shape[0] * target_width / img_frame.shape[1])
            row_imagined = resize_image(img_frame, (target_width, h))
        else:
            row_imagined = np.zeros((300, target_width, 3), dtype=np.uint8)
            if cv2 is not None:
                cv2.putText(row_imagined, "Coming soon", (target_width//2 - 100, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        row_imagined = np.ascontiguousarray(row_imagined)
        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.vstack([row_real, row_imagined])
        full_frame = np.ascontiguousarray(full_frame)
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video saved to: {save_path}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    yaml_mod = require_yaml()
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml_mod.load(f.read(), Loader=yaml_mod.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    yaml_mod = require_yaml()
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml_mod.load(f.read(), Loader=yaml_mod.FullLoader)
    return embodiment_args


def main(usr_args):
    yaml_mod = require_yaml()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml_mod.load(f.read(), Loader=yaml_mod.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root
    # The H200 eval path writes its own comparison video. RoboTwin's ffmpeg
    # eval-video pipe can fail on minimal system ffmpeg builds that do not
    # support libx264/-crf, then crash the rollout with BrokenPipeError.
    args["eval_video_log"] = False

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml_mod.load(f.read(), Loader=yaml_mod.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml_mod.load(f.read(), Loader=yaml_mod.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 10000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]

    WebsocketClientPolicy = import_policy_client()
    model = WebsocketClientPolicy(host=usr_args.get("host", "127.0.0.1"), port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=True,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def format_obs(observation, prompt, session_id=None):
    payload = {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right": observation["observation"]["right_camera"]["rgb"],
                "observation.state": observation["joint_action"]["vector"],
                "prompt": prompt,
                "annotation.task": prompt,
                "task": prompt,
            }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def _extract_action(response):
    if isinstance(response, dict):
        if "action" in response:
            return response["action"]
        if "actions" in response:
            return response["actions"]
    return response


def _normalize_joint_chunk(action):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2 or action.shape[-1] != 14:
        raise ValueError(f"Expected joint action chunk with shape (T, 14), got {action.shape}")
    return action

def add_eef_pose(new_pose, init_pose):
    if R is None:
        out_rot = quat_multiply_xyzw(init_pose[3:7], new_pose[3:7])
    else:
        new_pose_R = R.from_quat(new_pose[3:7][None])
        init_pose_R = R.from_quat(init_pose[3:7][None])
        out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])

def parse_bool_env(name, default=True):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no")

def safe_close_task_env(task_env, clear_cache=False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception as exc:
        print(f"warning: failed to close RoboTwin env cleanly: {exc}")

def clear_partial_robot(task_env):
    if hasattr(task_env, "robot"):
        try:
            delattr(task_env, "robot")
        except Exception as exc:
            print(f"warning: failed to clear partial robot state: {exc}")

def choose_episode_instruction(task_name, episode_info, test_num, instruction_type):
    episode_params = {}
    if isinstance(episode_info, dict):
        episode_params = episode_info.get("info", {})
    if not isinstance(episode_params, dict):
        episode_params = {}

    results = generate_episode_descriptions(task_name, [episode_params], test_num)
    choices = []
    if results:
        choices = results[0].get(instruction_type, [])
    if choices:
        return np.random.choice(choices)

    fallback = task_name.replace("_", " ")
    print(
        f"warning: no generated {instruction_type} instruction for task={task_name}; "
        f"using fallback prompt: {fallback}"
    )
    return fallback

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = parse_bool_env("EXPERT_CHECK", True)
    max_seed_attempts_raw = os.getenv("MAX_SEED_ATTEMPTS")
    max_seed_attempts = int(max_seed_attempts_raw) if max_seed_attempts_raw else max(test_num * 20, 100)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    seed_attempts = 0
    last_seed_error = None


    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if seed_attempts >= max_seed_attempts:
            raise RuntimeError(
                f"Failed to collect {test_num} valid eval seeds after "
                f"{seed_attempts} attempts. Last seed={now_seed - 1}, "
                f"last_error={last_seed_error}"
            )
        seed_attempts += 1
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        episode_info = None
        episode_metrics = {
            "task": task_name,
            "episode_index": now_id,
            "seed": now_seed,
            "client_gpu": os.getenv("CUDA_VISIBLE_DEVICES"),
            "episode_timeout_sec": EPISODE_TIMEOUT_SEC,
            "timeout": False,
            "timeout_reason": None,
            "success": False,
            "final_step": 0,
            "step_limit": None,
            "infer_count": 0,
            "action_steps": 0,
            "T_expert_filter": 0.0,
            "T_env_reset": 0.0,
            "T_generate_instruction": 0.0,
            "T_policy_reset": 0.0,
            "T_initial_obs": 0.0,
            "T_format_obs": 0.0,
            "T_policy_infer": 0.0,
            "T_take_action": 0.0,
            "T_get_obs": 0.0,
            "episode_elapsed": 0.0,
        }

        if expert_check:
            t_expert = time.perf_counter()
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                safe_close_task_env(TASK_ENV)
            except UnStableError as e:
                last_seed_error = f"UnStableError at seed={now_seed}: {e}"
                safe_close_task_env(TASK_ENV)
                clear_partial_robot(TASK_ENV)
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                last_seed_error = f"{type(e).__name__} at seed={now_seed}: {e}"
                safe_close_task_env(TASK_ENV)
                clear_partial_robot(TASK_ENV)
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue
            episode_metrics["T_expert_filter"] = time.perf_counter() - t_expert

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            last_seed_error = f"expert filter rejected seed={now_seed}"
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        t_env_reset = time.perf_counter()
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_metrics["T_env_reset"] = time.perf_counter() - t_env_reset
        episode_metrics["step_limit"] = int(TASK_ENV.step_lim)
        t_generate_instruction = time.perf_counter()
        if episode_info is None:
            episode_info = getattr(TASK_ENV, "info", {})
        instruction = choose_episode_instruction(args["task_name"], episode_info, test_num, instruction_type)
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction
        episode_metrics["T_generate_instruction"] = time.perf_counter() - t_generate_instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()
        session_id = f"{task_name}_ep{now_id:04d}_seed{now_seed}"
        episode_metrics["session_id"] = session_id
        episode_metrics["prompt"] = prompt
        t_policy_reset = time.perf_counter()
        model.reset({
            "session_id": session_id,
            "prompt": prompt,
            "task": task_name,
            "seed": now_seed,
            "episode_index": now_id,
        })
        episode_metrics["T_policy_reset"] = time.perf_counter() - t_policy_reset
        
        first = True
        full_obs_list = []
        gen_video_list = []
        full_action_history = []

        episode_start = time.perf_counter()
        timeout_sec = EPISODE_TIMEOUT_SEC if EPISODE_TIMEOUT_SEC > 0 else None

        def check_episode_timeout():
            if timeout_sec is None:
                return
            elapsed = time.perf_counter() - episode_start
            if elapsed > timeout_sec:
                raise EpisodeTimeoutError(
                    f"episode timeout after {elapsed:.1f}s > {timeout_sec:.1f}s"
                )

        try:
            t_initial_obs = time.perf_counter()
            initial_obs = TASK_ENV.get_obs()
            episode_metrics["T_initial_obs"] += time.perf_counter() - t_initial_obs
            inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
            [initial_obs['endpose']['left_gripper']] + \
            initial_obs['endpose']['right_endpose'] + \
            [initial_obs['endpose']['right_gripper']]
            inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
            t_format_obs = time.perf_counter()
            initial_formatted_obs = format_obs(initial_obs, prompt, session_id)
            episode_metrics["T_format_obs"] += time.perf_counter() - t_format_obs
            full_obs_list.append(initial_formatted_obs)
            first_obs = None
            while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
                check_episode_timeout()
                if first:
                    t_get_obs = time.perf_counter()
                    observation = TASK_ENV.get_obs()
                    episode_metrics["T_get_obs"] += time.perf_counter() - t_get_obs
                    t_format_obs = time.perf_counter()
                    first_obs = format_obs(observation, prompt, session_id)
                    episode_metrics["T_format_obs"] += time.perf_counter() - t_format_obs

                infer_payload = dict(first_obs)
                infer_payload.update({
                    "save_visualization": save_visualization,
                    "video_guidance_scale": video_guidance_scale,
                    "action_guidance_scale": action_guidance_scale,
                })
                t_policy_infer = time.perf_counter()
                ret = model.infer(infer_payload) #(TASK_ENV, model, observation)
                episode_metrics["T_policy_infer"] += time.perf_counter() - t_policy_infer
                episode_metrics["infer_count"] += 1
                check_episode_timeout()
                action = _extract_action(ret)
                if isinstance(ret, dict) and 'video' in ret:
                    imagined_video = ret['video']
                    gen_video_list.append(imagined_video)
                key_frame_list = []

                action = np.asarray(action)
                if action.ndim <= 2 or (action.ndim == 3 and action.shape[-1] == 14):
                    action_chunk = _normalize_joint_chunk(action)
                    for raw_action_step in action_chunk:
                        check_episode_timeout()
                        full_action_history.append(raw_action_step)
                        t_take_action = time.perf_counter()
                        TASK_ENV.take_action(raw_action_step, action_type='qpos')
                        episode_metrics["T_take_action"] += time.perf_counter() - t_take_action
                        episode_metrics["action_steps"] += 1
                        t_get_obs = time.perf_counter()
                        obs_raw = TASK_ENV.get_obs()
                        episode_metrics["T_get_obs"] += time.perf_counter() - t_get_obs
                        t_format_obs = time.perf_counter()
                        obs = format_obs(obs_raw, prompt, session_id)
                        episode_metrics["T_format_obs"] += time.perf_counter() - t_format_obs
                        full_obs_list.append(obs)
                        key_frame_list.append(obs)
                        episode_metrics["final_step"] = int(TASK_ENV.take_action_cnt)
                        if TASK_ENV.eval_success:
                            break
                else:
                    assert action.shape[2] % 4 == 0
                    action_per_frame = action.shape[2] // 4

                    start_idx = 1 if first else 0
                    for i in range(start_idx, action.shape[1]):
                        for j in range(action.shape[2]):
                            check_episode_timeout()
                            raw_action_step = action[:, i, j].flatten()
                            full_action_history.append(raw_action_step)

                            ee_action = action[:, i, j]
                            if action.shape[0] == 14:
                                ee_action = np.concatenate([
                                    ee_action[:3],
                                    euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                                    ee_action[6:10],
                                    euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                                    ee_action[13:14]
                                ])
                            elif action.shape[0] == 16:
                                ee_action =  add_init_pose(ee_action, inint_eef_pose)
                                ee_action = np.concatenate([
                                    ee_action[:3],
                                    ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                                    ee_action[7:11],
                                    ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                                    ee_action[15:16]
                                ])
                            else:
                                raise NotImplementedError
                            t_take_action = time.perf_counter()
                            TASK_ENV.take_action(ee_action, action_type='ee')
                            episode_metrics["T_take_action"] += time.perf_counter() - t_take_action
                            episode_metrics["action_steps"] += 1
                            episode_metrics["final_step"] = int(TASK_ENV.take_action_cnt)

                            if (j+1) % action_per_frame == 0:
                                t_get_obs = time.perf_counter()
                                obs_raw = TASK_ENV.get_obs()
                                episode_metrics["T_get_obs"] += time.perf_counter() - t_get_obs
                                t_format_obs = time.perf_counter()
                                obs = format_obs(obs_raw, prompt, session_id)
                                episode_metrics["T_format_obs"] += time.perf_counter() - t_format_obs
                                full_obs_list.append(obs)
                                key_frame_list.append(obs)
                            if TASK_ENV.eval_success:
                                break
                        if TASK_ENV.eval_success:
                            break

                first = False
                if key_frame_list:
                    first_obs = key_frame_list[-1]

                if TASK_ENV.eval_success:
                    succ = True
                    break
        except EpisodeTimeoutError as exc:
            succ = False
            episode_metrics["timeout"] = True
            episode_metrics["timeout_reason"] = str(exc)
            print(f"\033[91mEpisode timeout: {exc}\033[0m")

        episode_metrics["episode_elapsed"] = time.perf_counter() - episode_start
        episode_metrics["final_step"] = int(TASK_ENV.take_action_cnt)
        episode_metrics["success"] = bool(succ)
      

        if SAVE_COMPARISON_VIDEO:
            vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
            out_img_file = vis_dir / video_name
            save_comparison_video(
                real_obs_list=full_obs_list,
                imagined_video=None, #gen_video_list,
                action_history=full_action_history,
                save_path=str(out_img_file),
                fps=15 # Suggest adjusting fps based on simulation step
            )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        timing_file = Path(args['save_root']) / f'stseed-{st_seed}' / 'timings' / task_name / 'episodes.jsonl'
        append_jsonl(episode_metrics, timing_file)
        write_json({
          "succ_num": float(TASK_ENV.suc),
          "total_num": float(TASK_ENV.test_num),
          "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
          "last_episode": episode_metrics,
        }, out_json_file)
        
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--host", type=str, default="127.0.0.1", help='remote policy socket host.')
    parser.add_argument("--port", type=int, default=8000, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--test_num", type=int, default=100)
    args = parser.parse_args()

    yaml_mod = require_yaml()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml_mod.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    config.update({
        "host": args.host,
        "port": args.port,
        "save_root": args.save_root,
        "video_guidance_scale": args.video_guidance_scale,
        "action_guidance_scale": args.action_guidance_scale,
        "test_num": args.test_num,
    })

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    Sapien_TEST = import_robotwin_runtime()
    Sapien_TEST()
    main(usr_args)
