"""Video type: Trajectory Math trajectory mathematics video
Description: Generate a synthetic video of a red ball drawing mathematical expressions through its movement trajectory.
  - trajectory_math_complex_pemdas: 4-5 mathematical expressions for ball trajectory drawing (including +, -, x, / operations), the results need to be calculated according to the PEMDAS operation priority
  Each number and operator is drawn by a small ball along a predefined coordinate path"""

import os
import json
import random
import uuid
import shutil
import concurrent.futures
import math
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm

# --- 1. Configuration (global configuration) ---
plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

NUM_VIDEOS_TO_GENERATE = 500
NUM_WORKERS = 64

BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Trajectory_Math_v2/"
OUTPUT_VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SFT_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "sft.jsonl")
RL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "rl.jsonl")
METADATA_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata.jsonl")
VIDEO_PREFIX_IN_JSON = os.path.join(BASE_OUTPUT_DIR, "videos/")

FPS = 30
FRAMES_PER_CHAR = 60
PAUSE_BETWEEN_CHARS = 30
FINAL_PAUSE_FRAMES = 60

# --- 2. Character Path Data (unchanged) ---
CHAR_PATHS = {
    '0': [ [(0, 0), (0, 5), (3, 5), (3, 0), (0, 0)] ], '1': [ [(1.5, 0), (1.5, 5)] ],
    '2': [ [(0, 5), (3, 5), (3, 2.5), (0, 2.5), (0, 0), (3, 0)] ], '3': [ [(0, 5), (3, 5), (3, 2.5), (1.5, 2.5), (3, 2.5), (3, 0), (0, 0)] ],
    '4': [ [(0, 5), (0, 2.5), (3, 2.5), (3, 5), (3, 0)] ], '5': [ [(3, 5), (0, 5), (0, 2.5), (3, 2.5), (3, 0), (0, 0)] ],
    '6': [ [(3, 5), (0, 5), (0, 0), (3, 0), (3, 2.5), (0, 2.5)] ], '7': [ [(0, 5), (3, 5), (3, 0)] ],
    '8': [ [(1.5, 2.5), (0, 2.5), (0, 5), (3, 5), (3, 0), (0, 0), (0, 2.5), (3, 2.5)] ], '9': [ [(0, 0), (3, 0), (3, 5), (0, 5), (0, 2.5), (3, 2.5)] ],
    '+': [ [(0, 2.5), (3, 2.5)], [(1.5, 1), (1.5, 4)] ], '-': [ [(0, 2.5), (3, 2.5)] ],
    '=': [ [(0, 3.5), (3, 3.5)], [(0, 1.5), (3, 1.5)] ], 'x': [ [(0, 0), (3, 5)], [(0, 5), (3, 0)] ],
    '/': [ [(0, 5), (3, 0)] ]
}

# --- 3. Path Interpolation Helper (unchanged) ---
def _get_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        p1, p2 = np.array(path[i]), np.array(path[i+1])
        length += np.linalg.norm(p2 - p1)
    return length
CHAR_PATH_LENGTHS = {char: [_get_path_length(s) for s in strokes] for char, strokes in CHAR_PATHS.items()}
def _find_pos_on_path(path, progress, total_length):
    if progress <= 0.0: return path[0]
    if progress >= 1.0: return path[-1]
    target_dist = progress * total_length
    dist_so_far = 0
    for i in range(len(path) - 1):
        p1, p2 = np.array(path[i]), np.array(path[i+1])
        segment_len = np.linalg.norm(p2 - p1)
        if segment_len == 0: continue
        if dist_so_far + segment_len >= target_dist:
            dist_on_segment = target_dist - dist_so_far
            return tuple(p1 + (p2 - p1) * (dist_on_segment / segment_len))
        dist_so_far += segment_len
    return path[-1]

# --- 4. Numeric Options Helper (unchanged) ---
def generate_numeric_options(correct_answer, max_val):
    correct_answer = str(correct_answer)
    options = {correct_answer}
    while len(options) < 4:
        offset = random.randint(-10, 10)
        if offset == 0: offset = 10
        try: decoy = str(int(correct_answer) + offset)
        except ValueError: decoy = str(random.randint(0, 50))
        if decoy not in options: options.add(decoy)
        else: options.add(str(random.randint(max(int(correct_answer) - 50, -99), max_val + 50)))
    options = list(options)
    random.shuffle(options)
    return options

# --- 5. Challenge Generation Logic (unchanged) ---
def _generate_term():
    if random.random() < 0.6:
        if random.random() < 0.5:
            a, b = random.randint(2, 9), random.randint(2, 9)
            return (a * b, list(str(a)) + ['x'] + list(str(b)))
        else:
            b, a_mult = random.randint(2, 9), random.randint(2, 9)
            a = b * a_mult
            return (a // b, list(str(a)) + ['/'] + list(str(b)))
    else:
        a = random.randint(1, 25)
        return (a, list(str(a)))
def generate_challenge_data(challenge_id):
    NUM_TERMS = random.randint(4, 5)
    all_terms_values, all_term_chars, final_ops = [], [], []
    for i in range(NUM_TERMS):
        val, chars = _generate_term()
        all_terms_values.append(val)
        all_term_chars.append(chars)
        if i > 0: final_ops.append(random.choice(['+', '-']))
    final_result = all_terms_values[0]
    for i in range(1, NUM_TERMS):
        op, val = final_ops[i-1], all_terms_values[i]
        if op == '+': final_result += val
        else: final_result -= val
    problem_list = all_term_chars[0]
    for i in range(1, NUM_TERMS):
        problem_list.append(final_ops[i-1])
        problem_list.extend(all_term_chars[i])
    problem_list.append('=')
    correct_answer = str(final_result)
    options = generate_numeric_options(correct_answer, max_val=99)
    question = "A ball's trajectory draws a math problem, which can include operators like '+', '-', 'x' (multiply), and '/' (divide). Following the standard order of operations (PEMDAS/BODMAS), what is the final result?"
    video_filename = f"math_trajectory_{challenge_id}.mp4"
    return {"id": challenge_id, "challenge_type": "trajectory_math_complex_pemdas", "question": question, "options": options, "correct_answer_value": correct_answer, "correct_answer_label": chr(ord('A') + options.index(correct_answer)), "video_filename": video_filename, "video_filepath": os.path.join(OUTPUT_VIDEO_DIR, video_filename), "problem_list": problem_list}

# --- 6. Video Generation Engine ---
class VideoGenerator:
    ## --- MODIFICATION (Timestamp, Simplified) --- ##
    def __init__(self, challenge_data):
        self.data = challenge_data
        self.problem = self.data['problem_list']
        self.fig = plt.figure(figsize=(6, 6))
        self.fig.set_facecolor('white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.dynamic_patches = []
        self.total_frames = (FRAMES_PER_CHAR * len(self.problem)) + (PAUSE_BETWEEN_CHARS * len(self.problem)) + FINAL_PAUSE_FRAMES

        # --- New: Precompute "milliseconds (ms) status log" (only keep draw_char) ---
        self.events_timeline_ms = []
        current_time_ms = 0

        # Convert from frames to milliseconds
        draw_duration_ms = (FRAMES_PER_CHAR / FPS) * 1000
        pause_duration_ms = (PAUSE_BETWEEN_CHARS / FPS) * 1000
        final_pause_duration_ms = (FINAL_PAUSE_FRAMES / FPS) * 1000

        for i, char in enumerate(self.problem):
            # State 1: Draw
            start_ms = int(current_time_ms)
            current_time_ms += draw_duration_ms
            end_ms = int(current_time_ms)

            # --- Add only draw_char event ---
            self.events_timeline_ms.append({
                "event_type": "draw_char",
                "char": char,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

            # State 2: Paused (only advances time, does not record events)
            current_time_ms += pause_duration_ms

        # State 3: Final pause (only advances time, does not record events)
        current_time_ms += final_pause_duration_ms
        # --- End of modification ---
    ## --- MODIFICATION END --- ##

    def _draw_background(self):
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.set_xlim(0, 4); self.ax.set_ylim(0, 6)
        self.ax.axis('off')

    def _update(self, frame):
        # (This function remains unchanged)
        [p.remove() for p in self.dynamic_patches]; self.dynamic_patches = []
        self._draw_background()
        if frame >= self.total_frames: return
        frames_per_cycle = FRAMES_PER_CHAR + PAUSE_BETWEEN_CHARS
        char_index = frame // frames_per_cycle
        if char_index >= len(self.problem): return
        frame_in_cycle = frame % frames_per_cycle
        if frame_in_cycle < FRAMES_PER_CHAR:
            char_to_draw = self.problem[char_index]
            if char_to_draw not in CHAR_PATHS:
                print(f"Warning: Character '{char_to_draw}' not in CHAR_PATHS.")
                return
            strokes, path_lengths = CHAR_PATHS[char_to_draw], CHAR_PATH_LENGTHS[char_to_draw]
            num_strokes = len(strokes)
            if num_strokes == 0: return
            frames_per_stroke = FRAMES_PER_CHAR / num_strokes
            stroke_index = min(int(frame_in_cycle / frames_per_stroke), num_strokes - 1)
            frame_in_stroke = frame_in_cycle - (stroke_index * frames_per_stroke)
            denom = (frames_per_stroke - 1) if frames_per_stroke > 1 else 1
            progress = frame_in_stroke / denom
            path, path_len = strokes[stroke_index], path_lengths[stroke_index]
            if path_len == 0: return
            cx, cy = _find_pos_on_path(path, progress, path_len)
            self.dynamic_patches.append(self.ax.add_patch(patches.Circle((cx + 0.5, cy + 0.5), radius=0.15, facecolor='red', zorder=10)))

    def generate_video(self):
        # (This function remains unchanged)
        ani = animation.FuncAnimation(self.fig, self._update, frames=self.total_frames, interval=1000/FPS)
        writer = animation.writers['ffmpeg'](fps=FPS, metadata=dict(artist='Gemini'), bitrate=2000)
        try:
            ani.save(self.data['video_filepath'], writer=writer, dpi=120)
        finally:
            plt.close(self.fig)

# --- 7. Parallel Processing Worker ---
def process_single_challenge(challenge_id):
    try:
        challenge_data = generate_challenge_data(challenge_id)

        vg = VideoGenerator(challenge_data)
        vg.generate_video()

        # (Get self.events_timeline_ms status log)
        video_events_timeline_ms = vg.events_timeline_ms

        question_text = f"{challenge_data['question']}\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(challenge_data['options'])])
        video_server_path = os.path.join(VIDEO_PREFIX_IN_JSON, challenge_data['video_filename']).replace("\\", "/")

        sft = {"type":"intermedia_sft", "messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant.","loss":False}]},{"role":"user","content":[{"type":"video","uuid":str(uuid.uuid4()),"video":video_server_path},{"type":"text","text":question_text,"loss":False}]},{"role":"assistant","content":[{"type":"text","text":challenge_data['correct_answer_label'],"loss":True}]}],"extra":{"id":challenge_data['id'],"task":challenge_data['challenge_type'],"data_type":"video","data_source":"Synthetic_Video_Math_Trajectory","original_video_path":video_server_path}}
        rl = {"problem_id":challenge_data['id'],"problem":f"<video>\n{question_text}","data_type":"video","problem_type":challenge_data['challenge_type'],"videos":[video_server_path],"data_source":"Synthetic_Video_Math_Trajectory","answer":challenge_data['correct_answer_label']}

        # (The Metadata dictionary will now contain a shorter list of only draw_char events)
        metadata = {
            # RL-style format
            "problem_id": challenge_data['id'],
            "problem": f"<video>\n{question_text}",
            "data_type": "video",
            "problem_type": challenge_data['challenge_type'],
            "videos": [video_server_path],
            "data_source": "Synthetic_Video_Math_Trajectory",
            "answer": challenge_data['correct_answer_label'],

            # Detailed ground truth information for this script
            "ground_truth_details": {
                "problem_list": challenge_data['problem_list'],
                "question_data": {
                    "question": challenge_data['question'],
                    "options": challenge_data['options'],
                    "correct_answer_value": challenge_data['correct_answer_value']
                },

                # (Added "milliseconds" status log here - now only contains draw_char)
                "video_events_timeline_ms": video_events_timeline_ms
            }
        }

        return sft, rl, metadata
    except Exception as e:
        return (challenge_id, f"Error in processing: {e}\n{traceback.format_exc()}")

# --- 8. Main Execution Logic (unchanged) ---
def main():
    print(f"Starting dataset generation...\nBase output directory: {BASE_OUTPUT_DIR}\nNumber of videos: {NUM_VIDEOS_TO_GENERATE}\nWorkers: {NUM_WORKERS}")
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_challenge, i): i for i in range(NUM_VIDEOS_TO_GENERATE)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_VIDEOS_TO_GENERATE, desc="Generating Challenges"):
            results.append(future.result())

    print("\nWriting to JSONL files...")
    sft_recs, rl_recs, meta_recs, failed = [], [], [], []
    for res in results:
        if isinstance(res, tuple) and len(res) == 2:
            failed.append(res)
        elif res is not None:
            if len(res) >= 3:
                sft_recs.append(res[0]); rl_recs.append(res[1]); meta_recs.append(res[2])
            else:
                failed.append((-1, f"Worker returned incomplete data: {res}"))
        else:
            failed.append((-1, "Unknown error, result was None"))

    for path, data in zip([SFT_OUTPUT_FILE, RL_OUTPUT_FILE, METADATA_OUTPUT_FILE], [sft_recs, rl_recs, meta_recs]):
        with open(path, 'w') as f:
            for rec in data:
                f.write(json.dumps(rec) + '\n')

    print(f"\nSuccessfully generated {len(sft_recs)} challenges.")
    print(f"Videos: '{OUTPUT_VIDEO_DIR}/'\nSFT data: '{SFT_OUTPUT_FILE}'\nRL data: '{RL_OUTPUT_FILE}'\nMetadata: '{METADATA_OUTPUT_FILE}'")

    if failed:
        print(f"\nWARNING: {len(failed)} tasks failed.")
        for task_id, err in failed:
            print(f"  - Task {task_id}: {err}")

if __name__ == "__main__":
    main()
