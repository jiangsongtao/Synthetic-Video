"""
Video type: Atomic basic exercise video
Description: Generates composite videos of basic geometric shape motion, including the following subtypes:
- Bouncing_Counting: Count of objects bouncing between baffles
- Complex_Direction_Identification: Object composite direction motion recognition
- Trajectory_Shape_Identification: Object trajectory shape recognition (triangle/square/circle)
- Directional_Event_Counting: Statistics of the number of directional movements of objects
- Acceleration_Identification: Object acceleration/deceleration/uniform motion recognition
- Rotation_Direction: Object rotation direction identification (clockwise/counterclockwise)
- Rotation_Count: Counting the number of rotations of an object
"""

import os
import json
import random
import uuid # [--- ADDED ---]
import shutil # [--- ADDED ---]
import concurrent.futures
import math # [--- ADDED ---]
import traceback # [--- ADDED ---]

import matplotlib # [--- ADDED ---]
matplotlib.use('Agg') # [--- ADDED ---]
import matplotlib.pyplot as plt # [--- ADDED ---]
import matplotlib.patches as patches # [--- ADDED ---]
import matplotlib.animation as animation # [--- ADDED ---]
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

# --- Global configuration ---
CONFIG = {
    "output_dir": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/",
    "video_dir": "videos",
    # "qa_filename": "/cpfs02/user/songtao.jst/VideoData/synthetic_video_final/dataset.jsonl", # old SFT file
    "sft_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/sft.jsonl", # [--- MODIFIED ---]
    "metadata_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/metadata.jsonl", # [--- MODIFIED ---]
    "video_prefix_in_json": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/videos/", # [--- ADDED ---]
    "width": 512,
    "height": 512,
    "fps": 30,
    "bg_color": "black"
}

# --- Resource Library ---
SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue", "yellow", "purple", "cyan", "magenta", "orange", "white"]
ROTATION_SHAPES = ["arrow", "t_shape", "l_shape"]

# ---Auxiliary functions ---
def draw_shape(draw, shape_type, position, size, color, rotation=0):
    x, y = position
    if shape_type == "circle":
        points = [(size * np.cos(a), size * np.sin(a)) for a in np.linspace(0, 2*np.pi, 20)]
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
            r = size if i % 2 == 0 else size / 2
            points.append((r * np.cos(i * np.pi / 5 - np.pi/2), r * np.sin(i * np.pi / 5 - np.pi/2)))

    rotated_points = rotate_points(points, rotation)
    draw.polygon([(p[0] + x, p[1] + y) for p in rotated_points], fill=color)

def rotate_points(points, angle_rad):
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

# --- Task generator (refactored to return GT dictionary) ---

def generate_bouncing_counting(video_path, num_frames=None):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        shape, color, size = random.choice(SHAPES), random.choice(COLORS), 20
        target_bounce_count = random.randint(3, 7)
        pos = np.array([width / 2, height / 2], dtype=float)
        vel = np.array([random.uniform(-4, 4), random.choice([-10.0, 10.0])], dtype=float)
        paddle_height, top_paddle_y, bottom_paddle_y = 10, 40, height - 40
        bounces, frames_since_last_bounce, frames = 0, 15, []

        events_timeline_ms = [] # [--- MODIFIED ---]
        current_time_ms = 0

        while bounces < target_bounce_count:
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, top_paddle_y - paddle_height, width, top_paddle_y], fill="gray")
            draw.rectangle([0, bottom_paddle_y, width, bottom_paddle_y + paddle_height], fill="gray")
            draw_shape(draw, shape, tuple(pos), size, color)
            frames.append(np.array(img))

            pos += vel
            frames_since_last_bounce += 1
            current_time_ms += 1000.0 / fps # [--- MODIFIED ---]

            if pos[0] <= size or pos[0] >= width - size:
                vel[0] *= -1
            if (pos[1] <= top_paddle_y + size and vel[1] < 0) or (pos[1] >= bottom_paddle_y - size and vel[1] > 0):
                vel[1] *= -1
                if frames_since_last_bounce > 10:
                    bounces += 1
                    frames_since_last_bounce = 0

                    # [--- MODIFIED ---] Log events
                    events_timeline_ms.append({
                        "event_type": "bounce",
                        "bounce_count": bounces,
                        "timestamp_ms": int(current_time_ms)
                    })
            if len(frames) > 3000:
                break

        # ... (code to add extra frames) ...
        extra_frames = fps
        for _ in range(extra_frames):
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, top_paddle_y - paddle_height, width, top_paddle_y], fill="gray")
            draw.rectangle([0, bottom_paddle_y, width, bottom_paddle_y + paddle_height], fill="gray")
            draw_shape(draw, shape, tuple(pos), size, color)
            frames.append(np.array(img))
            pos += vel
            if pos[0] <= size or pos[0] >= width - size:
                vel[0] *= -1
            if (pos[1] <= top_paddle_y + size and vel[1] < 0) or (pos[1] >= bottom_paddle_y - size and vel[1] > 0):
                vel[1] *= -1

        safe_write_video(video_path, frames)

        question = "How many times does the object bounce off the gray boards?"
        correct_answer = bounces
        options = random.sample([i for i in range(1, 10) if i != correct_answer], 3) + [correct_answer]
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): str(opt) for i, opt in enumerate(options)},
            "correct_answer_value": str(correct_answer),
            "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "target_bounce_count": target_bounce_count
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_bouncing_counting: {e}")

def generate_complex_direction_identification(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 2: num_frames = 60

        shape, color, size = random.choice(SHAPES), random.choice(COLORS), 20
        directions = ["Right", "Left", "Up", "Down"]
        dir1, dir2 = random.sample(directions, 2)
        move_map = {"Right": (1, 0), "Left": (-1, 0), "Up": (0, -1), "Down": (0, 1)}
        vec1, vec2 = np.array(move_map[dir1]), np.array(move_map[dir2])
        move_dist = width / 4
        pos = np.array([width / 2, height / 2], dtype=float)
        frames = []
        mid_frame = num_frames // 2

        events_timeline_ms = [] # [--- MODIFIED ---]
        current_time_ms = 0.0

        start_pos1 = pos.copy()
        # [--- MODIFIED ---] Log event 1
        events_timeline_ms.append({
            "event_type": "move", "direction": dir1,
            "start_ms": int(current_time_ms),
            "end_ms": int(current_time_ms + (mid_frame / fps) * 1000)
        })
        for i in range(mid_frame):
            t = i / (mid_frame - 1) if mid_frame > 1 else 1.0
            current_pos = start_pos1 + vec1 * move_dist * t
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, tuple(current_pos), size, color)
            frames.append(np.array(img))
            current_time_ms += 1000.0 / fps # [--- MODIFIED ---]

        start_pos2 = start_pos1 + vec1 * move_dist
        # [--- MODIFIED ---] Log event 2
        events_timeline_ms.append({
            "event_type": "move", "direction": dir2,
            "start_ms": int(current_time_ms),
            "end_ms": int(current_time_ms + ((num_frames - mid_frame) / fps) * 1000)
        })
        for i in range(num_frames - mid_frame):
            t = i / (num_frames - mid_frame - 1) if (num_frames - mid_frame) > 1 else 1.0
            current_pos = start_pos2 + vec2 * move_dist * t
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, tuple(current_pos), size, color)
            frames.append(np.array(img))

        safe_write_video(video_path, frames)

        question = "What is the movement path of the object?"
        correct_answer = f"First to the {dir1.lower()}, then to the {dir2.lower()}"
        distractors = set()
        while len(distractors) < 3:
            d1, d2 = random.sample(directions, 2)
            distractor_str = f"First to the {d1.lower()}, then to the {d2.lower()}"
            if distractor_str != correct_answer:
                distractors.add(distractor_str)
        options = [correct_answer] + list(distractors)
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
            "correct_answer_value": correct_answer,
            "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "path": [dir1, dir2]
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_complex_direction_identification: {e}")

def generate_trajectory_shape_identification(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 2: num_frames = 60

        shape, color, size = "star", random.choice(COLORS), 10
        trajectory_shapes = ["Triangle", "Square", "Circle"]
        target_shape = random.choice(trajectory_shapes)
        center_x, center_y, radius = width / 2, height / 2, width / 3
        path_points = []

        events_timeline_ms = [] # [--- MODIFIED ---]

        if target_shape == "Circle":
            for i in range(num_frames):
                angle = 2 * np.pi * i / (num_frames - 1) if num_frames > 1 else 0
                path_points.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))
        else:
            if target_shape == "Square":
                points = [(center_x - radius, center_y - radius), (center_x + radius, center_y - radius),
                          (center_x + radius, center_y + radius), (center_x - radius, center_y + radius),
                          (center_x - radius, center_y - radius)]
            else: # Triangle
                points = [(center_x, center_y - radius), (center_x - radius, center_y + radius*0.8),
                          (center_x + radius, center_y + radius*0.8), (center_x, center_y - radius)]

            total_dist = sum(np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1))
            if total_dist == 0: total_dist = 1.0
            speed = total_dist / num_frames
            current_pos, target_node_idx = np.array(points[0]), 1
            path_points.append(tuple(current_pos))

            for _ in range(num_frames - 1):
                target_node = np.array(points[target_node_idx])
                direction_vec = target_node - current_pos
                dist_to_target = np.linalg.norm(direction_vec)
                if dist_to_target < speed:
                    current_pos, target_node_idx = target_node, (target_node_idx + 1) % len(points)
                else:
                    current_pos += (direction_vec / dist_to_target) * speed
                path_points.append(tuple(current_pos))

        # [--- MODIFIED ---] Log events
        events_timeline_ms.append({
            "event_type": "draw_trajectory",
            "shape": target_shape,
            "start_ms": 0,
            "end_ms": int((num_frames / fps) * 1000)
        })

        frames = []
        for pos in path_points:
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, pos, size, color)
            frames.append(np.array(img))

        safe_write_video(video_path, frames)

        question = "What geometric shape does the object's trajectory form?"
        options = random.sample([s for s in trajectory_shapes if s != target_shape], 2) + [target_shape]
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
            "correct_answer_value": target_shape,
            "correct_answer_label": chr(ord('A') + options.index(target_shape)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "trajectory_shape": target_shape
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_trajectory_shape_identification: {e}")

def generate_directional_event_counting(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 10: num_frames = 120

        shape, color, size = random.choice(SHAPES), random.choice(COLORS), 20
        move_directions = ["right", "left", "up", "down"]
        sequence = [random.choice(move_directions) for _ in range(random.randint(5, 8))]
        target_direction = random.choice(move_directions)
        pos = np.array([width / 2, height / 2], dtype=float)
        frames_per_move = max(10, num_frames // len(sequence))
        frames = []

        events_timeline_ms = [] # [--- MODIFIED ---]
        current_time_ms = 0.0

        for move in sequence:
            start_pos = pos.copy()
            move_vec = {"right": [1, 0], "left": [-1, 0], "up": [0, -1], "down": [0, 1]}[move]
            move_dist = 80

            # [--- MODIFIED ---] Log events
            start_ms = int(current_time_ms)
            end_ms = int(current_time_ms + (frames_per_move / fps) * 1000)
            events_timeline_ms.append({
                "event_type": "move", "direction": move,
                "start_ms": start_ms, "end_ms": end_ms
            })

            for i in range(frames_per_move):
                if i < frames_per_move - 5:
                    t = i / (frames_per_move - 6) if frames_per_move > 6 else 1.0
                    pos = start_pos + np.array(move_vec) * move_dist * t
                img = Image.new('RGB', (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                draw_shape(draw, shape, tuple(pos), size, color)
                frames.append(np.array(img))
                current_time_ms += 1000.0 / fps # [--- MODIFIED ---]

        safe_write_video(video_path, frames)

        correct_answer = sequence.count(target_direction)
        question = f"How many times did the object move to the {target_direction}?"
        options = random.sample([i for i in range(10) if i != correct_answer], 3) + [correct_answer]
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): str(opt) for i, opt in enumerate(options)},
            "correct_answer_value": str(correct_answer),
            "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "sequence": sequence, "target_direction": target_direction
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_directional_event_counting: {e}")

def generate_acceleration_identification(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 10: num_frames = 120

        shape, color, size = random.choice(SHAPES), random.choice(COLORS), 20
        motion_type = random.choice(["accelerating", "decelerating", "constant"])
        margin = size + 20
        setups = [
            {'start': [margin, height/2], 'axis': 0, 'dir': 1, 'name': 'Right'},
            {'start': [width-margin, height/2], 'axis': 0, 'dir': -1, 'name': 'Left'},
            {'start': [width/2, margin], 'axis': 1, 'dir': 1, 'name': 'Down'},
            {'start': [width/2, height-margin], 'axis': 1, 'dir': -1, 'name': 'Up'}
        ]
        setup = random.choice(setups)
        pos = np.array(setup['start'], dtype=float)

        if motion_type == "accelerating":
            speed, acceleration = 0.5, 0.25
        elif motion_type == "decelerating":
            speed, acceleration = 12.0, -0.25
        else:
            speed, acceleration = 7.0, 0.0

        frames = []

        # [--- MODIFIED ---] Log events
        events_timeline_ms = [{
            "event_type": "motion",
            "motion_type": motion_type,
            "direction": setup['name'],
            "start_ms": 0,
            "end_ms": int((num_frames / fps) * 1000)
        }]

        for _ in range(num_frames):
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, tuple(pos), size, color)
            frames.append(np.array(img))
            pos[setup['axis']] += speed * setup['dir']
            speed += acceleration
            if speed < 0:
                speed = 0
            if len(frames) >= num_frames:
                break

        if len(frames) < 10: # Make sure there are at least 10 frames
             frames.extend([frames[-1]] * (10 - len(frames)))

        safe_write_video(video_path, frames)

        question = "Is the object accelerating, decelerating, or moving at a constant speed?"
        option_map = {"accelerating": "Accelerating", "decelerating": "Decelerating", "constant": "Moving at a constant speed"}
        correct_answer_text = option_map[motion_type]
        options = list(option_map.values())
        random.shuffle(options)
        final_options = {chr(ord('A') + i): opt for i, opt in enumerate(options)}

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": final_options,
            "correct_answer_value": correct_answer_text,
            "correct_answer_label": [key for key, value in final_options.items() if value == correct_answer_text][0],
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "motion_type": motion_type
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_acceleration_identification: {e}")

def generate_rotation_direction(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 2: num_frames = 60

        shape, color, size = random.choice(ROTATION_SHAPES), random.choice(COLORS), 40
        direction = random.choice(["Clockwise", "Counter-clockwise"])
        rot_speed = (np.pi * 2 / fps) / 2
        if direction == "Clockwise":
            rot_speed *= -1

        # [--- MODIFIED ---] Log events
        events_timeline_ms = [{
            "event_type": "rotation",
            "direction": direction,
            "start_ms": 0,
            "end_ms": int((num_frames / fps) * 1000)
        }]

        frames = []
        for i in range(num_frames):
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, (width/2, height/2), size, color, rotation=i * rot_speed)
            frames.append(np.array(img))

        safe_write_video(video_path, frames)

        question = f"Is the {shape.replace('_', ' ')} spinning clockwise or counter-clockwise?"
        options = ["Clockwise", "Counter-clockwise", "It is not spinning"]
        correct_answer = direction
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): opt for i, opt in enumerate(options)},
            "correct_answer_value": correct_answer,
            "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "direction": direction
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_rotation_direction: {e}")

def generate_rotation_count(video_path, num_frames):
    try:
        width, height, fps, bg_color = CONFIG['width'], CONFIG['height'], CONFIG['fps'], CONFIG['bg_color']
        if num_frames < 2: num_frames = 60

        shape, color, size = random.choice(ROTATION_SHAPES), random.choice(COLORS), 40
        total_rotations = random.randint(2, 4)
        rot_speed = (np.pi * 2 * total_rotations) / num_frames

        # [--- MODIFIED ---] Log events
        events_timeline_ms = [{
            "event_type": "rotation",
            "total_rotations": total_rotations,
            "start_ms": 0,
            "end_ms": int((num_frames / fps) * 1000)
        }]

        frames = []
        for i in range(num_frames):
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            draw_shape(draw, shape, (width/2, height/2), size, color, rotation=i * rot_speed)
            frames.append(np.array(img))

        safe_write_video(video_path, frames)

        question = f"How many full rotations does the {shape.replace('_', ' ')} make?"
        correct_answer = total_rotations
        options = random.sample([i for i in range(1, 7) if i != correct_answer], 2) + [correct_answer]
        random.shuffle(options)

        # [--- MODIFIED ---] returns the GT dictionary
        return {
            "question": question,
            "options": {chr(ord('A') + i): str(opt) for i, opt in enumerate(options)},
            "correct_answer_value": str(correct_answer),
            "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
            "video_events_timeline_ms": events_timeline_ms,
            "ground_truth_details": {
                "shape": shape, "color": color, "total_rotations": total_rotations
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error in generate_rotation_count: {e}")

# global generator dictionary
GENERATORS = {
    "Bouncing_Counting": generate_bouncing_counting,
    "Complex_Direction_Identification": generate_complex_direction_identification,
    "Trajectory_Shape_Identification": generate_trajectory_shape_identification,
    "Directional_Event_Counting": generate_directional_event_counting,
    "Acceleration_Identification": generate_acceleration_identification,
    "Rotation_Direction": generate_rotation_direction,
    "Rotation_Count": generate_rotation_count,
}

# ---Unit of work for parallel processing (refactoring) ---
def process_single_video(args):
    try:
        gen_type, sample_index, video_dir = args
        generator_func = GENERATORS[gen_type]

        num_frames = None
        if gen_type != "Bouncing_Counting":
            duration_s = random.randint(4, 8)
            num_frames = duration_s * CONFIG['fps']

        video_filename = f"{gen_type}_{sample_index:04d}.mp4"
        video_filepath = os.path.join(video_dir, video_filename)

        # [--- MODIFIED ---]
        # generator_func now returns the complete GT dictionary
        gt_data = generator_func(video_filepath, num_frames=num_frames)

        # --- 1. Create SFT Entry (for dataset.jsonl) ---
        options_text = "\n".join([f"{key}. {value}" for key, value in sorted(gt_data["options"].items())])
        text_prompt = (f"Select the best answer to the following multiple-choice question based on the video. "
                       f"Respond with only the letter (A, B, C, D...) of the correct option. "
                       f"{gt_data['question']} Possible answer choices: {options_text}\nThe best answer is:\n  ")

        # [--- MODIFIED ---] uses os.path.abspath to match the original format
        sft_entry = {
            "id": f"{gen_type.lower()}_{sample_index}",
            "doc_id": sample_index,
            "task": gen_type,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video", "video": os.path.abspath(video_filepath)},
                    {"type": "text", "text": text_prompt}
                ]},
                {"role": "assistant", "content": gt_data["correct_answer_label"]}
            ]
        }

        # --- 2. Create Metadata Entry (for metadata.jsonl) ---
        video_server_path = os.path.join(CONFIG["video_prefix_in_json"], video_filename).replace("\\", "/")

        metadata_entry = {
            "problem_id": f"{gen_type.lower()}_{sample_index}",
            "problem": f"<video>\n{gt_data['question']}\n{options_text}",
            "data_type": "video",
            "problem_type": gen_type,
            "videos": [video_server_path],
            "data_source": "Synthetic_Video_Final",
            "answer": gt_data["correct_answer_label"],

            "ground_truth_details": {
                # [--- MODIFIED ---] moves the information in the GT dictionary here
                "question_data": {
                    "question": gt_data['question'],
                    "options": gt_data['options'],
                    "correct_answer_value": gt_data['correct_answer_value']
                },
                "video_events_timeline_ms": gt_data['video_events_timeline_ms'],
                "other_details": gt_data['ground_truth_details'] # (For example shape, color, etc.)
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
        for i in range(samples_per_type):
            tasks.append((gen_type, total_doc_id, video_dir))
            total_doc_id += 1

    print(f"Starting dataset generation with {max_workers} processes...")
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

    # [--- MODIFIED ---] writes two files
    # Write to SFT file
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

# [--- MODIFIED ---] matches the run parameters you provide
if __name__ == "__main__":
    generate_dataset(samples_per_type=500, max_workers=128)
