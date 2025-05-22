import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter

from controller.controller import ArmController

from inspire_hand import H1HandController
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml



RetargetingConfig.set_default_urdf_dir('/home/humanoid/avp_teleoperate/assets')
with Path('/home/humanoid/avp_teleoperate/assets/inspire_hand/inspire_hand.yml').open('r') as f:
    cfg = yaml.safe_load(f)
#print(f"{cfg=}")
left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
left_retargeting = left_retargeting_config.build()
right_retargeting = right_retargeting_config.build()
print("Hand retargeteting loaded")


def compute_hand_vector(left_hand_mat, right_hand_mat):
    """
    Converts a 25x3 hand keypoint matrix into a 6D joint control vector.

    Args:
        hand_array (np.ndarray): 25x3 matrix of hand keypoints (x, y, z).

    Returns:
        np.ndarray: 6D vector [pinky, ring, middle, index, thumb, thumb_angle].
    """
    tip_indices = [4, 9, 14, 19, 24]


    left_qpos = left_retargeting.retarget(left_hand_mat[tip_indices])[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    right_qpos = right_retargeting.retarget(right_hand_mat[tip_indices])[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    right_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
    right_angles.append(1.2 - right_qpos[8])
    right_angles.append(0.5 - right_qpos[9])

    left_angles = [1.7- left_qpos[i] for i in  [4, 6, 2, 0]]
    left_angles.append(1.2 - left_qpos[8])
    left_angles.append(0.5 - left_qpos[9])

    return left_angles, right_angles



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    args = parser.parse_args()
    print(f"args:{args}\n")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    #img_config = {'fps': 30, 'head_camera_type': 'realsense', 'head_camera_image_shape': [1080, 1920], 'head_camera_id_numbers': ['926522071700']}
    """img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        #'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_image_shape': [480, 848],  # Head camera resolution
        'head_camera_id_numbers': [4],
        #'wrist_camera_type': 'opencv',
        #'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        #'wrist_camera_id_numbers': [2, 4],
    }"""
    #img_config = {'fps': 30, 'head_camera_type': 'opencv', 'head_camera_image_shape': [240, 640], 'head_camera_id_numbers': [0]}
    img_config = {
        'fps':30,                                                          # frame per second
        'head_camera_type': 'realsense',                                  # opencv or realsense
        'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
        'head_camera_id_numbers': ["926522071700"],                       # realsense camera's serial number
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False

    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name,
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)

    # arm

    print('Initializing ArmController...')
    arm_ctrl = ArmController('./assets/h1_2/h1_2.urdf',
                             dt=1 / args.frequency,
                             vlim=1.0,
                             visualize=True)

    hand_ctrl = H1HandController()

    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False

    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        #if user_input.lower() == 'r':
        if user_input is not None:
            print("Start the subsequent program.")

            running = True
            while running:
                start_time = time.time()

                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()

                # send hand skeleton data to hand_ctrl.control_process
                lhand_vec, rhand_vec = compute_hand_vector(left_hand, right_hand)
                hand_ctrl.crtl(rhand_vec, lhand_vec)

                time_ik_start = time.time()
                arm_ctrl.left_ee_target_transformation = left_wrist
                arm_ctrl.right_ee_target_transformation = right_wrist
                arm_ctrl.control_step()

                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s') and args.record:
                    recording = not recording  # state flipping
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                    else:
                        recorder.save_episode()

                # record data
                if args.record:
                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    left_arm_state  = arm_ctrl.left_arm_q
                    right_arm_state = arm_ctrl.right_arm_q
                    left_arm_action = arm_ctrl.left_arm_action
                    right_arm_action = arm_ctrl.right_arm_action

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {
                                "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                                "qvel":   [],
                                "torque": [],
                            },
                            "right_arm": {
                                "qpos":   right_arm_state.tolist(),
                                "qvel":   [],
                                "torque": [],
                            },
                            "body": None,
                        }
                        actions = {
                            "left_arm": {
                                "qpos":   left_arm_action.tolist(),
                                "qvel":   [],
                                "torque": [],
                            },
                            "right_arm": {
                                "qpos":   right_arm_action.tolist(),
                                "qvel":   [],
                                "torque": [],
                            },
                            "body": None,
                        }
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                # print(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        arm_ctrl.goto_configuration(np.zeros_like(arm_ctrl.robot_model.q))
        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        print("Finally, exiting program...")
        exit(0)
