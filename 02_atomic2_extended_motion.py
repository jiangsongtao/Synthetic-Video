"""Video type: Atomic2 extended motion video
Description: Generate synthetic video of extended geometric shape motion, including the following subtypes:
  - Camera_Motion_Direction: simulate camera translation movement (up, down, left, right)
  - Relative_Direction: Identification of the relative movement direction of two objects (closer/away/parallel)
  - Relative_Speed_Comparison: Comparison of relative speeds of two objects
  - Event_Sequence: Recognition of the appearance/disappearance sequence of multiple shapes
  - Multi_Attribute_Change: Single object attribute change (color/size/shape/brightness)"""

import os
import json
import random
import uuid
import shutil
import concurrent.futures
import traceback # [--- ADDED ---]
import math # [--- ADDED ---]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageEnhance
import imageio # [--- ADDED ---]
import cv2 # [--- ADDED ---]

plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

# --- 1. Configuration (global configuration) ---

## --- MODIFICATION (User) --- ##
# (Updated to your "Atomic" path, consistent with the previous script)
## --------------------------- ##
CONFIG = {
    "output_dir": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/",
    "video_dir": "videos",
    "sft_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/sft.jsonl", # [--- MODIFIED ---]
    "metadata_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/metadata.jsonl", # [--- MODIFIED ---]
    "video_prefix_in_json": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/videos/", # [--- ADDED ---]
    "width": 512,
    "height": 512,
    "fps": 30,
    "bg_color": "black"
}
## --- MODIFICATION END --- ##


# --- Resource Library ---
SHAPES = ["circle", "square", "triangle", "star"]
COLORS = ["red", "green", "blue", "yellow", "purple", "cyan", "magenta", "orange", "white"]
ROTATION_SHAPES = ["arrow", "t_shape", "l_shape"]

# --- Auxiliary color mapping ---
COLOR_MAP = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "purple": (128, 0, 128), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "orange": (255, 165, 0), "white": (255, 255, 255),
    "black": (0, 0, 0)
}

# ---Auxiliary functions ---
def draw_shape(draw, shape_type, position, size, color, rotation=0):
    x, y = position; points = []
    if shape_type == "circle":
        draw.ellipse([(x - size, y - size), (x + size, y + size)], fill=color)
        return
    elif shape_type == "square":
        points = [(-size, -size), (size, -size), (size, size), (-size, size)]
    elif shape_type == "triangle":
        points = [(0, -size), (-size * 0.866, size * 0.5), (size * 0.866, size * 0.5)]
    elif shape_type == "arrow":
        points = [(0, -size), (size/2, 0), (size/4, 0), (size/4, size), (-size/4, size), (-size/4, 0), (-size/2, 0)]
    elif shape_type == "t_shape":
        points = [(-size, -size), (size, -size), (size, -size/2), (size/4, -size/2), (size/4, size), (-size/4, size), (-size/4, -size/2), (-size, -size/2)]
    elif shape_type == "l_shape":
        points = [(-size, -size), (-size/2, -size), (-size/2, size/2), (size, size/2), (size, size), (-size, size)]
    elif shape_type == "star":
        points = []
        for i in range(10):
            r = size if i % 2 == 0 else size / 2.2
            angle = i * np.pi / 5 - np.pi / 2
            points.append((r * np.cos(angle), r * np.sin(angle)))
    
    rotated_points = rotate_points(points, rotation)
    draw.polygon([(p[0] + x, p[1] + y) for p in rotated_points], fill=color)

def rotate_points(points, angle_rad):
    if angle_rad == 0: return points
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return [tuple(np.dot(p, rotation_matrix)) for p in points]

def safe_write_video(video_path, frames):
    """Safely write to video using OpenCV"""
    if not frames or len(frames) == 0:
        raise ValueError("Frames list is empty")
    
    valid_frames = []
    for f in frames:
        if f is not None:
            if not isinstance(f, np.ndarray):
                valid_frames.append(np.asarray(f))
            else:
                valid_frames.append(f)
    
    if len(valid_frames) == 0:
        raise ValueError("No valid frames after filtering")
    
    height, width = valid_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, CONFIG['fps'], (width, height))
    
    try:
        for frame in valid_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        out.release()

# --- ###################################################################### ---
# --- ###################### VIDEO GENERATOR FUNCTIONS ##################### ---
# --- ###################################################################### ---

## --- (all functions refactored to return GT dictionary and events_timeline_ms) --- ##

def generate_relative_direction(video_path, num_frames):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    obj1 = {"shape": random.choice(SHAPES), "color": random.choice(COLORS), "size": 20}
    obj2 = {"shape": random.choice(SHAPES), "color": random.choice(COLORS), "size": 20}
    while obj1['color'] == obj2['color']: obj2['color'] = random.choice(COLORS)
    
    direction_type = random.choice(["towards", "away", "parallel"])
    
    if direction_type == "towards":
        pos1, pos2 = np.array([width * 0.2, height / 2], dtype=float), np.array([width * 0.8, height / 2], dtype=float)
        vel1, vel2 = np.array([5, 0]), np.array([-5, 0])
    elif direction_type == "away":
        pos1, pos2 = np.array([width * 0.5 - 30, height / 2], dtype=float), np.array([width * 0.5 + 30, height / 2], dtype=float)
        vel1, vel2 = np.array([-5, 0]), np.array([5, 0])
    else:  # parallel
        pos1, pos2 = np.array([width * 0.2, height * 0.3], dtype=float), np.array([width * 0.2, height * 0.7], dtype=float)
        vel1, vel2 = np.array([5, 0]), np.array([5, 0])
        
    frames = []
    events_timeline_ms = [] # [--- MODIFIED ---]
    current_time_ms = 0.0
    
    # Animation set to half the total number of frames
    animation_frames = num_frames // 2
    
    events_timeline_ms.append({
        "event_type": "relative_motion",
        "motion_type": direction_type,
        "obj1": f"{obj1['color']} {obj1['shape']}",
        "obj2": f"{obj2['color']} {obj2['shape']}",
        "start_ms": 0,
        "end_ms": int((animation_frames / fps) * 1000)
    })
    
    collided = False
    for i in range(animation_frames):
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        draw_shape(draw, obj1['shape'], tuple(pos1), obj1['size'], COLOR_MAP[obj1['color']])
        draw_shape(draw, obj2['shape'], tuple(pos2), obj2['size'], COLOR_MAP[obj2['color']])
        frames.append(np.array(img))
        current_time_ms += 1000.0 / fps

        if direction_type == "towards" and not collided:
            distance = np.linalg.norm(pos1 - pos2)
            if distance <= (obj1['size'] + obj2['size']):
                collided = True
                vel1 = np.array([0, 0])
                vel2 = np.array([0, 0])
        
        pos1 += vel1
        pos2 += vel2
    
    # [--- MODIFIED ---] Use imageio.mimwrite instead of safe_write_video
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    
    question = f"How is the {obj1['color']} {obj1['shape']} moving in relation to the {obj2['color']} {obj2['shape']}?"
    q_map = {"towards": "Moving towards each other", "away": "Moving away from each other", "parallel": "Moving in parallel"}
    correct_answer = q_map[direction_type]
    distractors = [v for v in q_map.values() if v != correct_answer] + ["They are stationary"]
    options = [correct_answer] + distractors
    random.shuffle(options)
    
    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": {
            "obj1": obj1, "obj2": obj2, "motion_type": direction_type
        }
    }


def generate_camera_motion_direction(video_path, num_frames):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    world_w, world_h = width * 2, height * 2
    world_img = Image.new('RGB', (world_w, world_h), bg_color)
    draw = ImageDraw.Draw(world_img)
    objects = []
    for _ in range(50):
        pos = (random.randint(0, world_w), random.randint(0, world_h))
        shape = random.choice(SHAPES)
        color = random.choice(COLORS)
        size = random.randint(15, 30)
        draw_shape(draw, shape, pos, size, color)
        objects.append({"shape": shape, "color": color, "pos": pos})

    motion_type = random.choice(["pan_right", "pan_left", "pan_down", "pan_up"])
    
    frames = []
    events_timeline_ms = [] # [--- MODIFIED ---]
    events_timeline_ms.append({
        "event_type": "camera_motion",
        "motion_type": motion_type,
        "apparent_object_motion": {
            "pan_right": "Right to Left",
            "pan_left": "Left to Right",
            "pan_down": "Bottom to Top",
            "pan_up": "Top to Bottom"
        }[motion_type],
        "start_ms": 0,
        "end_ms": int((num_frames / fps) * 1000)
    })
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        if motion_type == "pan_right":
            box = (t * (world_w - width), (world_h - height)/2, t * (world_w - width) + width, (world_h + height)/2)
        elif motion_type == "pan_left":
            box = ((1-t) * (world_w - width), (world_h - height)/2, (1-t) * (world_w - width) + width, (world_h + height)/2)
        elif motion_type == "pan_down":
            box = ((world_w - width)/2, t * (world_h - height), (world_w + width)/2, t * (world_h - height) + height)
        elif motion_type == "pan_up":
            box = ((world_w - width)/2, (1-t) * (world_h - height), (world_w + width)/2, (1-t) * (world_h - height) + height)

        frame = world_img.crop(box).resize((width, height), Image.Resampling.LANCZOS)
        frames.append(np.array(frame))
        
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    
    question = "How are the objects in the scene moving?"
    q_map = {
        "pan_right": "Moving from right to left across the screen",
        "pan_left": "Moving from left to right across the screen",
        "pan_down": "Moving from bottom to top across the screen",
        "pan_up": "Moving from top to bottom across the screen"
    }
    correct_answer = q_map[motion_type]
    options = list(q_map.values())
    random.shuffle(options)
    
    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": {
            "motion_type": motion_type, "object_count": len(objects)
        }
    }

def generate_absolute_speed(video_path, num_frames=None):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    SPEED_MAP = {"slow": 60.0, "normal": 250.0, "fast": 1000.0}
    ANSWER_MAP = {"slow": "Slow motion", "normal": "Normal speed", "fast": "Time-lapse / Fast-forward"}
    speed_type = random.choice(list(SPEED_MAP.keys())); object_speed_pps = SPEED_MAP[speed_type]
    shape, color, size = random.choice(SHAPES), random.choice(COLORS), 20
    video_duration_s = 4.0; total_video_frames = int(video_duration_s * fps)
    pos = np.array([random.uniform(size * 2, width - size * 2), random.uniform(size * 2, height - size * 2)], dtype=float)
    angle = random.uniform(0, 2 * np.pi); velocity = np.array([np.cos(angle), np.sin(angle)]) * object_speed_pps
    frames = []; dt_per_frame = 1.0 / fps
    
    events_timeline_ms = [{
        "event_type": "motion",
        "speed_type": speed_type,
        "pixels_per_second": object_speed_pps,
        "start_ms": 0,
        "end_ms": int(video_duration_s * 1000)
    }]
    
    for _ in range(total_video_frames):
        img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
        draw_shape(draw, shape, tuple(pos), size, COLOR_MAP[color]); frames.append(np.array(img))
        pos += velocity * dt_per_frame
        if pos[0] <= size or pos[0] >= width - size: velocity[0] *= -1; pos[0] = np.clip(pos[0], size, width - size)
        if pos[1] <= size or pos[1] >= height - size: velocity[1] *= -1; pos[1] = np.clip(pos[1], size, height - size)
    
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    
    question = "How would you describe the speed of the video playback?"; correct_answer = ANSWER_MAP[speed_type]
    options = list(ANSWER_MAP.values()); random.shuffle(options)
    
    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": {
            "shape": shape, "color": color, "speed_type": speed_type, "pps": object_speed_pps
        }
    }

def generate_multi_attribute_change(video_path, num_frames):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    attr_to_change = random.choice(["color", "size", "shape", "brightness"])
    shape, color_name, size, position = random.choice(SHAPES), random.choice(COLORS), 30, (width/2, height/2)
    question, correct_answer, options = "", "", []
    
    events_timeline_ms = [] # [--- MODIFIED ---]
    gt_details = {"start_shape": shape, "start_color": color_name, "start_size": size}
    
    if attr_to_change == "color":
        start_color_name, end_color_name = random.sample([c for c in COLOR_MAP.keys() if c != 'black'], 2)
        start_rgb, end_rgb = COLOR_MAP[start_color_name], COLOR_MAP[end_color_name]
        question, correct_answer = f"The {shape} starts as {start_color_name}. What color does it become?", end_color_name.capitalize()
        distractors = random.sample([c.capitalize() for c in COLORS if c not in [start_color_name, end_color_name]], 3)
        options = [correct_answer] + distractors
        gt_details.update({"start_color": start_color_name, "end_color": end_color_name})
    elif attr_to_change == "size":
        change_type = random.choice(["growing", "shrinking"]); start_size, end_size = (10, 60) if change_type == "growing" else (60, 10)
        question, correct_answer = "What happens to the object's size?", f"It is {change_type}."
        distractors = [f"It is {'shrinking' if change_type == 'growing' else 'growing'}.", "It is pulsing.", "Its size is constant."]
        options = [correct_answer] + distractors
        gt_details.update({"start_size": start_size, "end_size": end_size, "change_type": change_type})
    elif attr_to_change == "shape":
        start_shape, end_shape = random.sample(SHAPES, 2)
        question, correct_answer = f"The object starts as a {start_shape}. What shape does it morph into?", end_shape.capitalize()
        distractors = random.sample([s.capitalize() for s in SHAPES if s not in [start_shape, end_shape]], 2) + ["It does not change shape"]
        options = [correct_answer] + distractors
        gt_details.update({"start_shape": start_shape, "end_shape": end_shape})
    elif attr_to_change == "brightness":
        change_type = random.choice(["brighter", "dimmer"]); start_factor, end_factor = (0.2, 1.0) if change_type == "brighter" else (1.0, 0.2)
        question, correct_answer = "How does the overall brightness of the video change?", f"It becomes {change_type}."
        distractors = [f"It becomes {'dimmer' if change_type == 'brighter' else 'brighter'}.", "It flashes.", "The brightness is constant."]
        options = [correct_answer] + distractors
        gt_details.update({"start_brightness": start_factor, "end_brightness": end_factor, "change_type": change_type})

    events_timeline_ms.append({
        "event_type": "attribute_change",
        "attribute": attr_to_change,
        "start_ms": 0,
        "end_ms": int((num_frames / fps) * 1000),
        **gt_details
    })

    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
        img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
        current_shape, current_color, current_size = shape, COLOR_MAP[color_name], size
        if attr_to_change == "color": current_color = tuple(int(start_rgb[c]*(1-t) + end_rgb[c]*t) for c in range(3))
        elif attr_to_change == "size": current_size = start_size*(1-t) + end_size*t
        elif attr_to_change == "shape": current_shape = start_shape if t < 0.5 else end_shape
        draw_shape(draw, current_shape, position, int(current_size), current_color)
        if attr_to_change == "brightness": img = ImageEnhance.Brightness(img).enhance(start_factor*(1-t) + end_factor*t)
        frames.append(np.array(img))
    
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    random.shuffle(options)
    
    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": gt_details
    }


def generate_event_sequence(video_path, num_frames):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    num_events = random.randint(3, 4); event_sequence = []
    while len(event_sequence) < num_events:
        obj = (random.choice(SHAPES), random.choice(COLORS), 25)
        if obj not in event_sequence: event_sequence.append(obj)
    
    event_duration_secs = 1.2; frames_per_event = int(event_duration_secs * fps)
    all_possible_positions = [(width*0.2, height/2), (width*0.4, height/2), (width*0.6, height/2), (width*0.8, height/2)]
    positions = all_possible_positions[:num_events]; random.shuffle(positions)
    event_type = random.choice(["appear", "disappear"]); frames = []
    
    events_timeline_ms = [] # [--- MODIFIED ---]
    current_time_ms = 0.0

    if event_type == "appear":
        question = "In what order do the shapes appear?"; visible_objects = []
        for event_idx in range(num_events):
            visible_objects.append((event_sequence[event_idx], positions[event_idx]))
            
            # [--- MODIFIED ---]
            start_ms = int(current_time_ms)
            end_ms = int(current_time_ms + (frames_per_event / fps) * 1000)
            events_timeline_ms.append({
                "event_type": "appear",
                "object": f"{event_sequence[event_idx][1]} {event_sequence[event_idx][0]}",
                "position": positions[event_idx],
                "start_ms": start_ms, "end_ms": end_ms
            })

            for _ in range(frames_per_event):
                img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
                for (shape, color, size), pos in visible_objects: draw_shape(draw, shape, pos, size, color)
                frames.append(np.array(img))
                current_time_ms += 1000.0 / fps # [--- MODIFIED ---]
    else:
        question = "In what order do the shapes disappear?"; visible_objects = [(event_sequence[i], positions[i]) for i in range(num_events)]
        for event_idx in range(num_events):
            # [--- MODIFIED ---]
            start_ms = int(current_time_ms)
            end_ms = int(current_time_ms + (frames_per_event / fps) * 1000)
            events_timeline_ms.append({
                "event_type": "disappear",
                "object": f"{event_sequence[event_idx][1]} {event_sequence[event_idx][0]}",
                "start_ms": start_ms, "end_ms": end_ms
            })
            
            for _ in range(frames_per_event):
                img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
                for (shape, color, size), pos in visible_objects: draw_shape(draw, shape, pos, size, color)
                frames.append(np.array(img))
                current_time_ms += 1000.0 / fps # [--- MODIFIED ---]
            
            obj_to_remove = event_sequence[event_idx]
            visible_objects = [vo for vo in visible_objects if vo[0] != obj_to_remove]
        
        for _ in range(frames_per_event):
            img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
            frames.append(np.array(img))
            
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    
    correct_answer = " -> ".join([f"the {o[1]} {o[0]}" for o in event_sequence]);
    distractors = set()
    while len(distractors) < 2:
        shuffled = random.sample(event_sequence, len(event_sequence)); distractor_str = " -> ".join([f"the {o[1]} {o[0]}" for o in shuffled])
        if distractor_str != correct_answer: distractors.add(distractor_str)
    options = [correct_answer] + list(distractors) + ["All shapes appear/disappear at once"]; random.shuffle(options)
    
    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": {
            "event_type": event_type, "sequence": [f"{o[1]} {o[0]}" for o in event_sequence]
        }
    }


def generate_relative_speed_comparison(video_path, num_frames):
    width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
    shape1, shape2 = random.sample(SHAPES, 2); color1, color2 = random.sample(COLORS, 2); size = 20
    speeds = [4.0, 8.0, 12.0]; random.shuffle(speeds); speed1, speed2 = speeds[0], speeds[1]
    pos1, pos2 = np.array([size+20, height*0.33], dtype=float), np.array([size+20, height*0.66], dtype=float)
    
    events_timeline_ms = [{
        "event_type": "relative_speed",
        "obj1": f"{color1} {shape1}", "speed1_pps": speed1,
        "obj2": f"{color2} {shape2}", "speed2_pps": speed2,
        "start_ms": 0, "end_ms": int((num_frames / fps) * 1000)
    }]
    
    frames = []
    for _ in range(num_frames):
        img = Image.new('RGB', (width, height), bg_color); draw = ImageDraw.Draw(img)
        draw_shape(draw, shape1, tuple(pos1), size, color1); draw_shape(draw, shape2, tuple(pos2), size, color2)
        frames.append(np.array(img)); pos1[0] += speed1; pos2[0] += speed2
        if pos1[0] > width or pos2[0] > width: break
        
    imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')
    
    question = "Which object is moving faster?";
    if speed1 > speed2: correct_answer = f"The {color1} {shape1}"; distractor_1 = f"The {color2} {shape2}"
    else: correct_answer = f"The {color2} {shape2}"; distractor_1 = f"The {color1} {shape1}"
    distractors = [distractor_1, "They are moving at the same speed", "They are not moving"]
    options = [correct_answer] + distractors; random.shuffle(options)

    return {
        "question": question,
        "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_events_timeline_ms": events_timeline_ms,
        "ground_truth_details": {
            "obj1": {"shape": shape1, "color": color1, "speed": speed1},
            "obj2": {"shape": shape2, "color": color2, "speed": speed2}
        }
    }


# --- (The following functions are not listed in GENERATORS and therefore remain as is without modification) ---
def generate_rotation_direction(video_path, num_frames):
    shape, color, size = random.choice(ROTATION_SHAPES), random.choice(COLORS), 40
    direction = random.choice(["Clockwise", "Counter-clockwise"]); rot_speed = (np.pi * 2 / CONFIG['fps']) / 2
    if direction == "Clockwise": rot_speed *= -1
    frames = []
    for i in range(num_frames):
        img = Image.new('RGB', (CONFIG['width'], CONFIG['height']), CONFIG['bg_color']); draw = ImageDraw.Draw(img)
        draw_shape(draw, shape, (CONFIG['width']/2, CONFIG['height']/2), size, color, rotation=i * rot_speed); frames.append(np.array(img))
    imageio.mimwrite(video_path, frames, fps=CONFIG['fps'], codec='libx264')
    question = f"Is the {shape.replace('_', ' ')} spinning clockwise or counter-clockwise?"; correct_answer = direction
    distractors = [d for d in ["Clockwise", "Counter-clockwise", "It is not spinning"] if d != correct_answer]
    options = [correct_answer] + distractors; random.shuffle(options)
    answer_key = chr(ord('A') + options.index(correct_answer))
    return {"question": question, "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)}, "answer": answer_key}

def generate_rotation_count(video_path, num_frames):
    shape, color, size = random.choice(ROTATION_SHAPES), random.choice(COLORS), 40
    total_rotations = random.randint(2, 4); rot_speed = (np.pi * 2 * total_rotations) / num_frames if num_frames > 0 else 0; frames = []
    for i in range(num_frames):
        img = Image.new('RGB', (CONFIG['width'], CONFIG['height']), CONFIG['bg_color']); draw = ImageDraw.Draw(img)
        draw_shape(draw, shape, (CONFIG['width']/2, CONFIG['height']/2), size, color, rotation=i * rot_speed); frames.append(np.array(img))
    final_rotation = (num_frames - 1) * rot_speed if num_frames > 0 else 0
    for _ in range(CONFIG['fps']):
        img = Image.new('RGB', (CONFIG['width'], CONFIG['height']), CONFIG['bg_color']); draw = ImageDraw.Draw(img)
        draw_shape(draw, shape, (CONFIG['width']/2, CONFIG['height']/2), size, color, rotation=final_rotation); frames.append(np.array(img))
    imageio.mimwrite(video_path, frames, fps=CONFIG['fps'], codec='libx264')
    question = f"How many full rotations does the {shape.replace('_', ' ')} make?"; correct_answer = total_rotations
    distractor_pool = [i for i in range(max(1, correct_answer - 2), correct_answer + 3) if i != correct_answer]
    distractors = random.sample(distractor_pool, min(2, len(distractor_pool)))
    options = [correct_answer] + distractors; random.shuffle(options)
    answer_key = chr(ord('A') + options.index(correct_answer))
    return {"question": question, "options": {chr(ord('A') + i): str(opt) for i, opt in enumerate(options)}, "answer": answer_key}

def generate_action_identification(video_path, num_frames):
    actions = ["orbiting", "pulsing", "vertical_bouncing", "horizontal_shaking", "figure_eight", "spiraling", "color_cycling",  "colliding"]
    chosen_action = random.choice(actions)
    obj1_shape, obj2_shape = random.sample(SHAPES, 2); obj1_color, obj2_color = random.sample(COLORS, 2)
    obj1_size, obj2_size = random.randint(15, 30), random.randint(15, 30)
    frames = []; margin = int(CONFIG['width'] * 0.3)
    ref_pos = np.array([random.randint(margin, CONFIG['width'] - margin), random.randint(margin, CONFIG['height'] - margin)], dtype=float)
    pos1, pos2 = ref_pos.copy(), np.array([random.randint(50, 462), random.randint(50, 462)], dtype=float)
    if chosen_action == "vertical_bouncing": vel1 = np.array([0, random.uniform(8, 12)])
    else: vel1 = np.array([random.uniform(-4, 4), random.uniform(5, 10)])
    vel2 = np.array([random.uniform(-4, 4), random.uniform(-4, 4)]); spiral_direction = ""
    if chosen_action == "spiraling": spiral_direction = random.choice(["inward", "outward"])
    cycle_colors_rgb = []
    if chosen_action == "color_cycling":
        color_names = random.sample([c for c in COLORS if c != obj1_color], random.randint(3, 4))
        cycle_colors_rgb = [COLOR_MAP[c] for c in color_names]
    figure_eight_orientation = "";
    if chosen_action == "figure_eight": figure_eight_orientation = random.choice(['horizontal', 'vertical'])
    orbiter_obj = None
    if chosen_action == "orbiting": orbiter_obj = 'obj1' if random.random() > 0.5 else 'obj2'
    if chosen_action == "chasing":
        pos1 = np.array([random.randint(50, 462), random.randint(50, 462)], dtype=float); pos2 = np.array([random.randint(50, 462), random.randint(50, 462)], dtype=float)
        vel2 = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
    if chosen_action == "colliding":
        pos1 = np.array([random.uniform(50, 150), random.uniform(50, 462)], dtype=float); pos2 = np.array([random.uniform(362, 462), random.uniform(50, 462)], dtype=float)
        if np.linalg.norm(pos2 - pos1) > 0:
            direction = (pos2 - pos1) / np.linalg.norm(pos2 - pos1); vel1 = direction * random.uniform(5, 8); vel2 = -direction * random.uniform(5, 8)
    for i in range(num_frames):
        img = Image.new('RGB', (CONFIG['width'], CONFIG['height']), CONFIG['bg_color']); draw = ImageDraw.Draw(img); t = i / (num_frames - 1)
        if chosen_action == "orbiting":
            orbit_radius = random.uniform(CONFIG['width'] / 5, CONFIG['width'] / 3); angle = 2 * np.pi * t; stationary_pos = ref_pos
            orbiter_pos = (stationary_pos[0] + orbit_radius * np.cos(angle), stationary_pos[1] + orbit_radius * np.sin(angle))
            if orbiter_obj == 'obj1':
                draw_shape(draw, obj1_shape, orbiter_pos, obj1_size, COLOR_MAP[obj1_color]); draw_shape(draw, obj2_shape, stationary_pos, obj2_size, COLOR_MAP[obj2_color])
            else:
                draw_shape(draw, obj1_shape, stationary_pos, obj1_size, COLOR_MAP[obj1_color]); draw_shape(draw, obj2_shape, orbiter_pos, obj2_size, COLOR_MAP[obj2_color])
        elif chosen_action == "chasing":
            pos2 += vel2
            if pos2[0] <= obj2_size or pos2[0] >= CONFIG['width'] - obj2_size: vel2[0] *= -1
            if pos2[1] <= obj2_size or pos2[1] >= CONFIG['height'] - obj2_size: vel2[1] *= -1
            trail_distance = (obj1_size + obj2_size) * 2.5; target_velocity_norm = np.linalg.norm(vel2)
            trail_point = pos2 - (vel2 / target_velocity_norm) * trail_distance if target_velocity_norm > 0 else pos2
            chaser_speed = 5.0; chaser_direction_vec = trail_point - pos1; chaser_distance_to_target = np.linalg.norm(chaser_direction_vec)
            if chaser_distance_to_target > 1: pos1 += (chaser_direction_vec / chaser_distance_to_target) * chaser_speed
            draw_shape(draw, obj1_shape, tuple(pos1), obj1_size, COLOR_MAP[obj1_color]); draw_shape(draw, obj2_shape, tuple(pos2), obj2_size, COLOR_MAP[obj2_color])
        elif chosen_action == "colliding":
            pos1 += vel1; pos2 += vel2
            if np.linalg.norm(pos1-pos2) < (obj1_size + obj2_size):
                norm_pos_diff = np.linalg.norm(pos1-pos2)**2
                if norm_pos_diff > 0:
                    v1_new = vel1 - 2 * np.dot(vel1-vel2, pos1-pos2) / norm_pos_diff * (pos1-pos2); v2_new = vel2 - 2 * np.dot(vel2-vel1, pos2-pos1) / norm_pos_diff * (pos2-pos1); vel1, vel2 = v1_new, v2_new
            draw_shape(draw, obj1_shape, tuple(pos1), obj1_size, COLOR_MAP[obj1_color]); draw_shape(draw, obj2_shape, tuple(pos2), obj2_size, COLOR_MAP[obj2_color])
        else: # Single-object actions
            current_pos, current_size, current_color = pos1, obj1_size, COLOR_MAP[obj1_color]
            if chosen_action == "pulsing":
                scale_factor = 1.0 + 0.75 * np.sin(t * random.randint(3, 5) * np.pi); current_size = obj1_size * scale_factor; current_pos = ref_pos
            elif chosen_action == "vertical_bouncing":
                pos1 += vel1
                if pos1[1] <= obj1_size or pos1[1] >= CONFIG['height'] - obj1_size: vel1[1] *= -1
                current_pos = tuple(pos1)
            elif chosen_action == "horizontal_shaking":
                offset = random.randint(25, 50) * np.sin(t * random.randint(8, 12) * np.pi); current_pos = (ref_pos[0] + offset, ref_pos[1])
            elif chosen_action == "figure_eight":
                path_width, path_height = random.uniform(CONFIG['width']/4, CONFIG['width']/2.5), random.uniform(CONFIG['height']/5, CONFIG['height']/3); angle = t * 2 * np.pi
                dx = path_width * np.sin(angle); dy = path_height * np.sin(2 * angle)
                if figure_eight_orientation == 'vertical': dx, dy = dy, dx # Swap components
                current_pos = (ref_pos[0] + dx, ref_pos[1] + dy)
            elif chosen_action == "spiraling":
                num_rotations, max_radius = random.randint(3, 5), random.uniform(CONFIG['width']/3, CONFIG['width']/2.2); angle = t * num_rotations * 2 * np.pi
                radius = max_radius * (1 - t) if spiral_direction == "inward" else max_radius * t
                current_pos = (ref_pos[0] + radius * np.cos(angle), ref_pos[1] + radius * np.sin(angle))
            elif chosen_action == "color_cycling":
                current_pos = ref_pos; color_idx = int(t * len(cycle_colors_rgb) * random.randint(3, 5)) % len(cycle_colors_rgb); current_color = cycle_colors_rgb[color_idx]
            draw_shape(draw, obj1_shape, current_pos, current_size, current_color)
        frames.append(np.array(img))
    imageio.mimwrite(video_path, frames, fps=CONFIG['fps'], codec='libx264')
    q_obj1, q_obj2 = f"the {obj1_color} {obj1_shape}", f"the {obj2_color} {obj2_shape}"; action_map = {"orbiting_obj1_orbits_obj2": (f"What is {q_obj1} doing in relation to {q_obj2}?", f"Orbiting {q_obj2}"), "orbiting_obj2_orbits_obj1": (f"What is {q_obj2} doing in relation to {q_obj1}?", f"Orbiting {q_obj1}"), "pulsing": (f"What action is {q_obj1} performing?", "Pulsing in size"), "vertical_bouncing": (f"What is the main activity of {q_obj1}?", "Bouncing up and down"), "horizontal_shaking": (f"What is {q_obj1} doing?", "Shaking left and right"), "figure_eight": (f"What path is {q_obj1} tracing?", "Moving in a figure-eight pattern"), "spiraling_inward": (f"How is {q_obj1} moving?", "Spiraling inward"), "spiraling_outward": (f"How is {q_obj1} moving?", "Spiraling outward"), "color_cycling": (f"What is the main activity of {q_obj1}?", "Cycling through different colors"), "chasing": (f"What is {q_obj1} doing in relation to {q_obj2}?", f"Chasing {q_obj2}"), "colliding": (f"What is the primary interaction between the two objects?", "Moving towards each other and colliding")}
    if chosen_action == "spiraling": question, correct_answer = action_map[f"spiraling_{spiral_direction}"]
    elif chosen_action == "orbiting": question, correct_answer = action_map[f"orbiting_{orbiter_obj}_orbits_{'obj2' if orbiter_obj == 'obj1' else 'obj1'}"]
    else: question, correct_answer = action_map[chosen_action]
    all_answers = [v[1] for v in action_map.values()]; exclusions = {correct_answer}
    if correct_answer == "Moving in a figure-eight pattern": exclusions.add("Shaking left and right")
    elif correct_answer == "Shaking left and right": exclusions.add("Moving in a figure-eight pattern")
    distractor_pool = [ans for ans in all_answers if ans not in exclusions]; num_distractors = min(3, len(distractor_pool)); distractors = random.sample(distractor_pool, num_distractors)
    options = [correct_answer] + distractors; random.shuffle(options)
    answer_key = chr(ord('A') + options.index(correct_answer))
    return {"question": question, "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)}, "answer": answer_key}
# --- Global generator dictionary ---
GENERATORS = {
    "Camera_Motion_Direction": generate_camera_motion_direction, 
    "Relative_Direction": generate_relative_direction, 
    "Relative_Speed_Comparison": generate_relative_speed_comparison, 
    "Event_Sequence": generate_event_sequence,
    "Multi_Attribute_Change": generate_multi_attribute_change,
    # "Absolute_Speed": generate_absolute_speed, # [--- ADDED ---]
    # "Trajectory_Shape_Identification": generate_trajectory_shape_identification, # [--- ADDED ---]
    # "Directional_Event_Counting": generate_directional_event_counting, # [--- ADDED ---]
    # "Acceleration_Identification": generate_acceleration_identification, # [--- ADDED ---]
    # "Rotation_Count": generate_rotation_count, # [--- ADDED ---]
    # "Action_Identification": generate_action_identification, # (As per your original code, this is commented out)
    # "Bouncing_Counting": generate_bouncing_counting, # (As per your original code, this is commented out)
    # "Rotation_Direction": generate_rotation_direction, # (As per your original code, this is commented out)
}

# --- Main function and parallel processing ---
## --- MODIFICATION (v1.9 - unified format) --- ##
def process_single_video(args):
    try:
        gen_type, sample_index, video_dir = args
        generator_func = GENERATORS[gen_type]

        tasks_that_control_their_own_duration = ["Event_Sequence", "Absolute_Speed", "Bouncing_Counting"]
        num_frames = None
        if gen_type not in tasks_that_control_their_own_duration:
            duration_s = random.randint(4, 8)
            num_frames = duration_s * CONFIG['fps'] 

        video_filename = f"{gen_type}_{sample_index:04d}.mp4"
        video_filepath = os.path.join(video_dir, video_filename)
        
        # [--- MODIFIED ---] 
        # generator_func now returns the full GT dictionary
        gt_data = generator_func(video_filepath, num_frames=num_frames)
        
        # --- 1. Create SFT Entry (for sft.jsonl) ---
        options_text = "\n".join([f"{key}. {value}" for key, value in sorted(gt_data["options"].items())])
        text_prompt = (f"Select the best answer to the following multiple-choice question based on the video. "
                       f"Respond with only the letter (A, B, C, D...) of the correct option. "
                       f"{gt_data['question']} Possible answer choices: \n{options_text}\nThe best answer is:")
        
        sft_entry = {
            "id": f"{gen_type.lower()}_{sample_index}", 
            "doc_id": sample_index, 
            "task": gen_type,
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "video", "video": os.path.abspath(video_filepath)}, {"type": "text", "text": text_prompt}]},
                {"role": "assistant", "content": gt_data["correct_answer_label"]}]}
        
        # --- 2. Create Metadata Entry (for metadata.jsonl) ---
        video_server_path = os.path.join(CONFIG["video_prefix_in_json"], video_filename).replace("\\", "/")
        
        metadata_entry = {
            # Unified RL-style format
            "problem_id": f"{gen_type.lower()}_{sample_index}",
            "problem": f"<video>\n{gt_data['question']}\n{options_text}",
            "data_type": "video",
            "problem_type": gen_type,
            "videos": [video_server_path], # Make sure it's a list
            "data_source": "Synthetic_Video_Atomic", # [--- MODIFIED ---]
            "answer": gt_data["correct_answer_label"],
            
            # Unified Ground Truth format
            "ground_truth_details": {
                "question_data": {
                    "question": gt_data['question'],
                    "options": gt_data['options'],
                    "correct_answer_value": gt_data['correct_answer_value']
                },
                "video_events_timeline_ms": gt_data['video_events_timeline_ms'],
                "other_details": gt_data['ground_truth_details'] # (e.g. shape, color, etc.)
            }
        }
        
        return sft_entry, metadata_entry # [--- MODIFIED ---]
    
    except Exception as e:
        print(f"Error processing {gen_type}_{sample_index}: {str(e)}\n{traceback.format_exc()}")
        return None

# --- Main generating function (refactored) ---
def generate_dataset(samples_per_type=1500, max_workers=128):
    output_dir = CONFIG['output_dir']
    video_dir = os.path.join(output_dir, CONFIG['video_dir'])
    sft_path = CONFIG['sft_filename'] # [--- MODIFIED ---]
    metadata_path = CONFIG['metadata_filename'] # [--- MODIFIED ---]
    os.makedirs(video_dir, exist_ok=True)

    tasks = []
    total_doc_id = 0
    for gen_type in GENERATORS.keys():
        print(f"Queueing {samples_per_type} tasks for type: {gen_type}")
        for i in range(samples_per_type):
            tasks.append((gen_type, total_doc_id, video_dir)); total_doc_id += 1

    print(f"\nStarting dataset generation with {max_workers} processes...")
    successful_sft_entries = []
    successful_meta_entries = [] # [--- MODIFIED ---]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Generating videos"):
            try:
                result = future.result()
                if result is not None:
                    sft_entry, metadata_entry = result
                    successful_sft_entries.append(sft_entry)
                    successful_meta_entries.append(metadata_entry) # [--- MODIFIED ---]
            except Exception as e:
                print(f"Task failed with exception: {e}")
    
    with open(sft_path, 'w', encoding='utf-8') as f:
        for entry in successful_sft_entries:
            f.write(json.dumps(entry) + '\n')
            
    # Write to Metadata file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in successful_meta_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nDataset generation complete!")
    print(f"Total videos generated: {len(successful_sft_entries)} / {len(tasks)}")
    print(f"Success rate: {len(successful_sft_entries) / len(tasks) * 100:.1f}%")
    print(f"SFT Dataset saved to: {sft_path}")
    print(f"Metadata Dataset saved to: {metadata_path}")

if __name__ == "__main__":
    generate_dataset(samples_per_type=200, max_workers=128)
