import os
import json
import tqdm
import random
import numpy as np
import cv2
import pickle
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from loguru import logger

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import truncnorm
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import math
import copy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd


def transform_images(text: str) -> str:
    occurrences = re.findall(r"<image>\n", text)
    
    if len(occurrences) <= 1:
        return text
    
    count = 0
    def replace_func(match):
        nonlocal count
        count += 1
        return f"Image-{count}: <image>\n"

    transformed_text = re.sub(r"<image>\n", replace_func, text)
    
    return transformed_text



def is_image_black(image_array):
    # Check if all pixel values are 0 (black image)
    return np.all(image_array == 0)

def is_image_almost_black(image_array, threshold=0.80, tolerance=10):
    """
    Check if the RGB image is almost entirely black.

    Parameters:
    image_array (ndarray): The input RGB image array of shape (height, width, 3).
    threshold (float): The fraction of pixels that must be close to black for the image to be considered almost black.
    tolerance (int): The maximum value a pixel can have in each channel to be considered black (0-255 scale).

    Returns:
    bool: True if the image is almost black, False otherwise.
    """
    # Check if all channels are below the tolerance for each pixel
    nearly_black_pixels = np.sum(np.all(image_array <= tolerance, axis=-1))
    total_pixels = image_array.shape[0] * image_array.shape[1]

    fraction_black = nearly_black_pixels / total_pixels

    return fraction_black >= threshold

def is_image_path_almost_black(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return False
    image_array = np.array(img)
    return is_image_almost_black(image_array)


def is_image_path_black(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return False
    image_array = np.array(img)
    return np.all(image_array == 0)

def compare_images(image1_path, image2_path):
    # Load the images
    try:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
    except:
        return -1
    
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    if is_image_almost_black(img1_array) or is_image_almost_black(img2_array):
        return -1
    
    difference = np.abs(img1_array - img2_array)
    
    num_different_pixels = np.sum(difference > 0)
    
    total_pixels = img1_array.size
    
    percentage_difference = (num_different_pixels / total_pixels) * 100
    
    return percentage_difference

def denormalize_with_params(normalized_data, params):
    """
    使用参数将归一化的数据还原到原始范围
    
    Args:
        normalized_data: 归一化后的任意维度数组，但最后一个维度必须是7
        params: 归一化参数字典
        
    Returns:
        numpy.ndarray: 还原后的数据，维度与输入一致
    """
    # 转换为numpy数组
    data_array = np.array(normalized_data)
    original_shape = data_array.shape
    
    # 检查最后一个维度是否为7
    if len(original_shape) < 2:
        raise ValueError(f"输入数据至少应该是2维的，但得到的是{len(original_shape)}维")
    
    if original_shape[-1] != 7:
        raise ValueError(f"输入数据的最后一个维度应该是7，但得到的是{original_shape[-1]}")
    
    # 将数据reshape为(..., 7)的形状，然后flatten前面的维度
    working_data = data_array.reshape(-1, 7)
    
    # 提取参数
    scale = np.array(params['scale'])
    offset = np.array(params['offset'])
    
    # 反向变换: x = normalized * scale + offset
    original_data = working_data * scale + offset
    
    # 恢复原始形状
    original_data = original_data.reshape(original_shape)
    
    return original_data

def agibot_process(action,action_type='action'):
    if action_type=='fast-chunk':
        return action
    def convert_to_int(element):
        if isinstance(element, list):
            return [convert_to_int(sub_element) for sub_element in element]
        else:
            try:
                return int(element)
            except ValueError:
                return element  # Return the element unchanged if it can't be converted

    return convert_to_int(action)


def songling_process_single(action,action_type='action'):
    if type(action) is list:
        for i in range(3,6):
            if abs(action[i])>180000:
                action[i]=((action[i] + 180000) % 360000) - 180000
        if action_type=='fast-chunk':
            new_action=action[:-1]+[int(action[-1]/1000)]
        else:
            new_action=[int(temp/1000.0) for temp in action]
        if action_type=='action' or action_type=='fast-chunk':
            if new_action[-1]<0:
                new_action[-1]=0
            elif new_action[-1]>=1:
                new_action[-1]=1
            else:
                new_action[-1]=0
        if action_type=='state':
            new_action[-1]=round((new_action[-1]//10000)/7.0,1)
        return new_action
    else:
        return None

def songling_process(action, action_type='action'):
    if type(action) is list:
        if len(action) == 7 and all(not isinstance(item, list) for item in action):
            return songling_process_single(action, action_type)
        else:
            new_action = []
            for one_action in action:
                processed = songling_process(one_action, action_type)
                if processed is not None:
                    new_action.append(processed)
            return new_action if new_action else None
    else:
        return None


def format_songling(action):
    if action is None:
        return 'None'
    elif type(action) is list:
        if type(action[0]) is list:
            if len(action)==2:
                return "\nleft: <action> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </action> \nright: <action> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </action>".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[1][0],action[1][1],action[1][2],action[1][3],action[1][4],action[1][5],action[1][6])
            elif len(action)==1:
                return "{{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}}".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6])
            else:
                return 'None'
        else:
            return "{{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}}".format(action[0],action[1],action[2],action[3],action[4],action[5],action[6])
    else:
        return 'End-Effector with units in mm and degrees'

def normalize_angle(angle):
    if abs(angle)>180:
        return (angle + 180) % 360 - 180
    else:
        return angle


def format_songling_v2(action,state=False,data_type=None):
    if action is None:
        return 'None'
    elif type(action) is list:
        if type(action[0]) is list:
            pass
        else:
            action = [action]
        if state:
            pass
        else:
            for i in range(len(action)):
                action[i][3]=normalize_angle(action[i][3])
                action[i][4]=normalize_angle(action[i][4])
                action[i][5]=normalize_angle(action[i][5])
        if len(action)==2:
            if state:
                return "\nleft: <position> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </position> \nright: <position> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </position>".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[1][0],action[1][1],action[1][2],action[1][3],action[1][4],action[1][5],action[1][6])
            else:
                return "\nleft: <action> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </action> \nright: <action> {{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}} </action>".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[1][0],action[1][1],action[1][2],action[1][3],action[1][4],action[1][5],action[1][6])
        elif len(action)==1:
            return "{{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}}".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6])
        else:
            return 'None'
    else:
        if data_type:
            return f'{data_type} End-Effector with units in mm and degrees'
        return 'End-Effector with units in mm and degrees'

def format_songling_simple(action,state=False,data_type=None):
    if action is None:
        return 'None'
    elif type(action) is list:
        if type(action[0]) is list:
            pass
        else:
            action = [action]
        if state:
            pass
        else:
            for i in range(len(action)):
                action[i][3]=normalize_angle(action[i][3])
                action[i][4]=normalize_angle(action[i][4])
                action[i][5]=normalize_angle(action[i][5])
        if len(action)==2:
            if state:
                return "\nleft: <(position)> ({} {} {} {} {} {} {}) </(position)> \nright: <(position)> ({}, {}, {}, {}, {}, {}, {}) </(position)>".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[1][0],action[1][1],action[1][2],action[1][3],action[1][4],action[1][5],action[1][6])
            else:
                return "\nleft: <(action)> ({} {} {} {} {} {} {}) </(action)> \nright: <(action)> ({}, {}, {}, {}, {}, {}, {}) </(action)>".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[1][0],action[1][1],action[1][2],action[1][3],action[1][4],action[1][5],action[1][6])
        elif len(action)==1:
            return "({} {} {} {} {} {} {})".format(action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6])
        else:
            return 'None'
    else:
        if data_type:
            return f'{data_type} End-Effector hide units in mm and degrees within () format'
        return 'End-Effector hide units in mm and degrees within () format'

def trojectory_example_prompt(images,task):
    prompt=f"<trajectory> <task> {task} </task>:"
    t_len=len(images)-1
    for i,one in enumerate(range(len(images))):
        temp_p=int((i/t_len)*100)
        prompt=prompt+f" {temp_p}% <image>\n"
    prompt=prompt+'</trajectory>'
    return prompt


def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        # {-1: "close gripper", 0: None, 1: "open gripper"},
        {0: "close gripper", 1: "open gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != -2:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        description = "stop"

    return description

def denoise_action(action):
    xyz = action[:3]
    rpy = action[3:6]
    open_val = action[6]

    def normalize_angle(angle):
        return (angle + 180) % 360 - 180

    def process_dims(values, ref_max=None):
        abs_values = [abs(v) for v in values]
        max_val = max(abs_values)
        
        if max_val == 0:
            return [0, 0, 0]
            
        processed = [0, 0, 0]
        main_idx = abs_values.index(max_val)
        processed[main_idx] = 1 if values[main_idx] > 0 else -1
        
        for i in range(3):
            if i == main_idx:
                continue
            threshold = max_val * 0.25
            if abs(values[i]) < threshold:
                processed[i] = 0
            else:
                processed[i] = 1 if values[i] > 0 else (-1 if values[i] < 0 else 0)
        return processed

    adjusted_rpy = [normalize_angle(v) for v in rpy]
    
    xyz_processed = process_dims(xyz)
    max_xyz = max(abs(v) for v in xyz)
    
    rpy_abs = [abs(v) for v in adjusted_rpy]
    max_rpy = max(rpy_abs)
    
    if max_rpy < max_xyz * 0.25 and max_rpy <= 4:
        rpy_processed = [0, 0, 0]
    else:
        rpy_processed = process_dims(adjusted_rpy)

    if max_xyz < max_rpy * 0.25 and max_xyz <= 5:
        xyz_processed = [0, 0, 0]
    else:
        pass
    return xyz_processed + rpy_processed + [open_val]

def describe_action(action,threshold=0.3,denoise=True):
    if denoise:
        action=denoise_action(action)
    else:
        for i in range(len(action)-1):
            if action[i]<-threshold:
                action[i]=-1
            elif action[i]>threshold:
                action[i]=1
            else:
                action[i]=0

    return describe_move(action), action

def format_songling_think_one(action,threshold=0.3,denoise=True):
    #think_threshold dis when denoise true
    if action is None:
        return 'None'
    elif type(action) is list:
        if type(action[0]) is list:
            if len(action)==2:
                return f"left: <action> {describe_action(action[0],threshold,denoise)[0]} </action> \nright: <action> {describe_action(action[1],threshold,denoise)[0]} </action>"
            elif len(action)==1:
                return f"{describe_action(action[0],threshold,denoise)[0]}"
            else:
                return 'None'
        else:
            return f"{describe_action(action,threshold,denoise)[0]}"
    else:
        return 'None'

def format_songling_think(action,threshold=0.3,denoise=True,multi=False):
    if multi:
        pass
    else:
        action = [action]
    if len(action)==1:
        return f"<think> {format_songling_think_one(action[0],threshold,denoise)} </think> "
    action_think_str=''
    for i,one in enumerate(action):
        action_think_str+=f'{i+1}. '+format_songling_think_one(one,threshold,denoise)+'\n'
    return f"<think> {action_think_str} </think> "

def bridge_action_preprocess(one,data,key,td=1):
    trajectory_id,step_id,step_num,view_id= key.split('-')
    step_id=int(step_id)
    step_num=int(step_num)
    next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
    next_one=data.get(next_key)
    if td>1:
        temp_one=data.get(f'{trajectory_id}-{step_id+td-1}-{step_num}-{view_id}')
    else:
        temp_one=one
    if next_one is None:
        return None
    if temp_one is None:
        return None
    one_state=one['position_7d']
    next_state=next_one['position_7d']
    action=[next_state[i]-one_state[i] for i in range(6)]
    action+=[round(temp_one['action'][-1])]
    return action

def droid_action_preprocess(one,data,key,td=1):
    trajectory_id,step_id,step_num,view_id= key.split('-')
    step_id=int(step_id)
    step_num=int(step_num)
    next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
    next_one=data.get(next_key)
    if next_one is None:
        return None
    one_state=one['position_7d']
    next_state=next_one['position_7d']
    action=[next_state[i]-one_state[i] for i in range(6)]
    open_action=next_state[-1]-one_state[-1]
    if open_action>0:
        open_action=1
    elif open_action<0:
        open_action=0
    else:
        open_action=1 if next_state[-1] >=0.85 else 0
    action+=[open_action]
    return action

def bridge_position_preprocess(one=None,data=None,key=None,td=1,action=True):
    if action:
        trajectory_id,step_id,step_num,view_id= key.split('-')
        step_id=int(step_id)
        step_num=int(step_num)
        next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
        next_one=data.get(next_key)
        if next_one is None:
            return None
        position=next_one['position_7d']
    else:
        position=one['position_7d']
    if type(position) is list:
        processed_position = [
            max(min(int(position[i] * 1000), 999), -999) if i < 3 
            else int(position[i] * 360 / 3.1416)
            for i in range(6)
        ]
        processed_position+=[int(position[-1])*100]
        return processed_position
    else:
        return None

def songling_position_preprocess(one=None,data=None,key=None,td=1,action=True):
    if action:
        trajectory_id,step_id,step_num,view_id= key.split('-')
        step_id=int(step_id)
        step_num=int(step_num)
        next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
        next_one=data.get(next_key)
        if next_one is None:
            return None
        position=next_one['position_7d']
    else:
        position=one['position_7d']
    if type(position) is list:
        if type(position[0]) is list:
            new_position=[]
            for one_position in position:
                processed_position = [int(temp/1000.0) for temp in position]
                new_position.append(processed_position)
            return new_position
        else:
            processed_position = [int(temp/1000.0) for temp in position]
            return processed_position
    else:
        return None

def agibot_position_preprocess(one=None,data=None,key=None,td=1,action=True):
    if action:
        trajectory_id,step_id,step_num,view_id= key.split('-')
        step_id=int(step_id)
        step_num=int(step_num)
        next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
        next_one=data.get(next_key)
        if next_one is None:
            return None
        position=next_one['position_7d']
    else:
        position=one['position_7d']
    if type(position) is list:
        if type(position[0]) is list:
            new_position=[]
            for one_position in position:
                processed_position = [int(temp) for temp in one_position]
                processed_position[2] = processed_position[2]-200
                new_position.append(processed_position)
            return new_position
        else:
            processed_position = [int(temp) for temp in position]
            processed_position[2] = processed_position[2]-200
            return processed_position
    else:
        return None

def default_position_preprocess(one=None,data=None,key=None,td=1,action=True):
    if action:
        trajectory_id,step_id,step_num,view_id= key.split('-')
        step_id=int(step_id)
        step_num=int(step_num)
        next_key=f'{trajectory_id}-{step_id+td}-{step_num}-{view_id}'
        next_one=data.get(next_key)
        if next_one is None:
            return None
        position=next_one['position_7d']
    else:
        position=one['position_7d']
    return position

def default_position_process(position,action_type='action'):
    return position

def default_action_preprocess(one,data=None,key=None,td=1):
    return one['action']

def default_process(action,action_type='action'):
    if action_type=='fast-chunk':
        return action
    return [int(temp) for temp in action]

def droid_process_single(action,action_type='action'):
    if action_type=='fast-chunk':
        return action
    if type(action) is list:
        processed_action = [
            max(min(int(action[i] * 1000), 999), -999) if i < 3 
            else int(action[i] * 360 / 3.1416)
            for i in range(6)
        ]
        processed_action+=[action[-1]]
        return processed_action
    else:
        return None

def droid_process(action,action_type='action'):
    if type(action) is list:
        if type(action[0]) is list:
            new_action=[]
            for one_action in action:
                new_action.append(droid_process_single(one_action,action_type))
            return new_action
        else:
            return droid_process_single(action,action_type)
    else:
        return None


class DataProcessor():
    def __init__(self):
        self.action_process={
            'songling':songling_process,
            'agibot':agibot_process,
            'bridge':droid_process,
            'droid':droid_process,
            'default':default_process
        }
        self.action_format={
            'songling':format_songling_v2,
            'agibot':format_songling_v2,
            'bridge':format_songling_v2,
            'droid':format_songling_v2,
            'default':format_songling_v2
        }
        self.action_format_simple={
            'songling':format_songling_simple,
            'agibot':format_songling_simple,
            'bridge':format_songling_simple,
            'droid':format_songling_simple,
            'default':format_songling_simple
        }
        self.action_think_format={
            'songling':format_songling_think,
            'agibot':format_songling_think,
            'bridge':format_songling_think,
            'droid':format_songling_think,
            'default':format_songling_think
        }
        self.action_preprocess={
            'songling':default_action_preprocess,
            'agibot':default_action_preprocess,
            'bridge':bridge_action_preprocess,
            'droid':droid_action_preprocess,
            'default':default_action_preprocess
        }
        self.position_preprocess={
            'songling':songling_position_preprocess,
            'agibot':agibot_position_preprocess,
            'bridge':bridge_position_preprocess,
            'droid':bridge_position_preprocess,
            'default':default_position_preprocess
        }
        self.position_process={
            'songling':default_position_process,
            'agibot':default_position_process,
            'bridge':default_position_process,
            'droid':default_position_process,
            'default':default_position_process
        }

        self.image_prompt_templete={
            1:'<image>\n',
            2:'Image-1: <image>\nImage-2: <image>\n',
            3:'Image-1: <image>\nImage-2: <image>\nImage-3: <image>\n'}
        self.system_prompt='You are a visual-language assistant designed to interpret spatial and task-related information from images and text. Provide precise, context-aware responses and actionable guidance to assist in achieving task objectives.'
        self.prompt_templete={
            "v3":"Image-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image.\nPlease directly rate score following below rules:\nPositive Score: If the second image is closer to achieving task objectives than the first image, assign a positive score based on the significance of the improvement.\nNegative Score: If the second image deviates further from the task objectives compared to the first image, assign a negative score based on the degree of deterioration.\nZero Score: If both images demonstrate the same level of task completion, assign a score of 0.\nThe task needs to accomplish is: <task> {} </task> <score>",

            "v3_think":"0% <image>\nThis image is the trajectory beginning of the following two images\nImage-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image.\nPlease directly rate score following below rules:\nPositive Score: If the second image is closer to achieving task objectives than the first image, assign a positive score based on the significance of the improvement.\nNegative Score: If the second image deviates further from the task objectives compared to the first image, assign a negative score based on the degree of deterioration.\nZero Score: If both images demonstrate the same level of task completion, assign a score of 0.\nThe task needs to accomplish is: <task> {} </task> <score>",

            'think':" Please give reasoning process enclosed within <think> reasoning process here </think>.",

            "vqa":"{}",

            "task_vqa":"Image-1: <image>\nImage-2: <image>\nCompare two images and infer what kind of task is achieving. <infer> <task>",

            "task_done":"The 1 means yes, the 0 means no. Check if the robot has completed its task: <task> {} </task> <done>",

            "context_task_done":"<goal> <task> {} </task>: <image>\n<goal>\n<image>\n The 1 means yes, the 0 means no. Refer to the goal, check if the robot has completed its task: <task> {} </task> <done>",

            "image_done":"<goal> <image>\n<goal>\n<image>\n The 1 means yes, the 0 means no. Check if the robot has completed its image goal <done>",

            "action_inverse":"Image-1: <image>\nImage-2: <image>\nCompare two images and infer what action between them. <infer> <action>",

            "task_action":"The current position state of the robotic arm's end gripper in the image is as follows: <state> {} </state>. What action should the robot take to get better completion of instruction: <task> {} </task> <action>",

            "fast_action":"The current position state of the robotic arm's end gripper in the image is as follows: <state> {} </state>. What action chunks should the robot take to get better completion of instruction: <task> {} </task> <chunk>",

            "task_action_simple":"The current position state of the robotic arm's end gripper in the image is as follows: <state> {} </state>. What action should the robot take to get better completion of instruction: <task> {} </task> <(action)>",

            "task_position":"The current position state of the robotic arm's end gripper in the image is as follows: <position> {} </position>. What position should the robot take to get better completion of instruction: <task> {} </task> <position>",

            "task_position_simple":"The current position state of the robotic arm's end gripper in the image is as follows: <(position)> {} </(position)>. What position should the robot take to get better completion of instruction: <task> {} </task> <(position)>",

            "task_action_score":"The current position state of the robotic arm's end gripper in the image is as follows: <state> {} </state>. The action robot take now: <action> {} </action>. Please rate score of the action for achieving task: <task> {} </task> <score>",

            'action':"{{x: {}mm, y: {}mm, z: {}mm, roll: {} degrees, pitch: {} degrees, yaw: {} degrees, open: {}}}"}
        self.answer_templete={
            "v3":"{}",
            "vqa":"{}",
            "task_vqa":"{}",
            "task_done":"{}",
            "action_inverse":"{}",
            "task_action":"{}",
            "task_action_score":"{}"
        }