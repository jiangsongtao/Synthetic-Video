"""
Video type: Chip & Cup chip cup tracking video
Description: Generate a tracking video of casino chips added/removed between two cups (A and B)
- chip_cup_tracking: Track the final chips in the target cup after 8-12 add/remove operations
Chip value: 5, 10, 20, 50, 100, 500 (different colors)
"""

import os
import json
import random
import uuid
import shutil
import concurrent.futures
import traceback # [--- FIX 1 of 2: 添加了 import ---]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm


plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

NUM_VIDEOS_TO_GENERATE = 500
NUM_WORKERS = 64 # Adjust according to your CPU core number

BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Chip_Cup_v2_EN/"
OUTPUT_VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SFT_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "sft.jsonl")
RL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "rl.jsonl")
METADATA_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata.jsonl")
VIDEO_PREFIX_IN_JSON = os.path.join(BASE_OUTPUT_DIR, "videos/")

FPS = 30
INITIAL_REVEAL_FRAMES = 150 # Initial display (5 seconds)
OP_ANIMATION_FRAMES = 60    # Animation for each operation (2 seconds)
OP_STATIC_FRAMES = 30       # Pause between each operation (1 second)
FINAL_PAUSE_FRAMES = 60     # final pause

# --- 2. Chip & Cup Helpers (chip and cup auxiliary functions) ---

# [--- NEW ---] defines chip pool and cup
# Chip (ID, display value, color)
CHIP_POOL = [
    ('c5_1', '5', '#FFFFFF'),  # White
    ('c5_2', '5', '#FFFFFF'),
    ('c10', '10', '#E63946'),  # Red
    ('c20', '20', '#2A9D8F'),  # Green
    ('c50_1', '50', '#457B9D'), # Blue
    ('c50_2', '50', '#457B9D'),
    ('c100_1', '100', '#000000'),# Black
    ('c100_2', '100', '#000000'),
    ('c500', '500', '#E9C46A')  # Yellow
]

CUP_IDS = ['A', 'B'] # We will use 2 cups

def format_chip_id(chip_id):
    """ 'c50_1' -> '50' """
    return chip_id.split('_')[0][1:]

def format_chips_for_answer(chip_list):
    """ ['c50_1', 'c20'] -> '20, 50' """ # Note: sorted
    if not chip_list:
        return "Empty"
    # Sort by chip value to make the answer unique
    sorted_chips = sorted([format_chip_id(c) for c in chip_list], key=int)
    return ", ".join(sorted_chips)

def generate_answer_options(correct_chips, d1_initial, d2_swapped, d3_action_bug):
    """Generate 4 smart distraction options"""
    correct_answer = format_chips_for_answer(correct_chips)
    options = {correct_answer}

    options.add(format_chips_for_answer(d1_initial))
    if len(options) < 4:
        options.add(format_chips_for_answer(d2_swapped))
    if len(options) < 4:
        options.add(format_chips_for_answer(d3_action_bug))

    # complete
    deck = random.sample(CHIP_POOL, 5)
    while len(options) < 4:
        decoy_pile = [c[0] for c in random.sample(deck, 3)]
        options.add(format_chips_for_answer(decoy_pile))

    options = list(options)
    random.shuffle(options)
    return options, correct_answer


# --- 3. Simulation Engine (core: truth simulator) ---

def simulate_operations(initial_states, operations):
    """
Simulate all operations and return the final state.
    initial_states = {'A': ['c50_1'], 'B': ['c100_1']}
    operations = [('add', 'A', 'c20'), ('remove', 'B', 'c100_1')]
    """
    final_states = {cup: chips.copy() for cup, chips in initial_states.items()}

    for op_type, cup_id, chip_id in operations:
        if op_type == 'add':
            final_states[cup_id].append(chip_id)
        elif op_type == 'remove':
            if chip_id in final_states[cup_id]:
                final_states[cup_id].remove(chip_id)

    return final_states

# --- 4. Challenge Generation Logic (Core: Challenge Generator) ---

def generate_challenge_data(challenge_id):

    num_ops = random.randint(8, 12)

    # 1. Initialize cups and chips
    available_chips = [c[0] for c in CHIP_POOL]
    random.shuffle(available_chips)

    initial_states = {cup_id: [] for cup_id in CUP_IDS}

    # 2. Allocate initial chips
    num_initial_chips = random.randint(3, 5)
    for _ in range(num_initial_chips):
        chip = available_chips.pop()
        cup = random.choice(CUP_IDS)
        initial_states[cup].append(chip)

    # 3. Generate operation sequence
    operations = []
    sim_states = {cup: chips.copy() for cup, chips in initial_states.items()}

    for _ in range(num_ops):
        target_cup = random.choice(CUP_IDS)

        # Decide whether 'add' or 'remove'
        if not sim_states[target_cup]:
            op_type = 'add'
        elif not available_chips:
            op_type = 'remove'
        else:
            op_type = random.choice(['add', 'remove'])

        # perform operations
        if op_type == 'add':
            chip_to_add = available_chips.pop()
            sim_states[target_cup].append(chip_to_add)
            operations.append(('add', target_cup, chip_to_add))
        else: # op_type == 'remove'
            chip_to_remove = random.choice(sim_states[target_cup])
            sim_states[target_cup].remove(chip_to_remove)
            available_chips.append(chip_to_remove) # Put the chips back into the pool
            operations.append(('remove', target_cup, chip_to_remove))

    # --- Run a true simulation ---
    final_states = simulate_operations(initial_states, operations)

    # --- Identify the problem ---
    target_cup_id = random.choice(CUP_IDS)
    question = (
        f"The video begins by revealing the initial chips in each cup. "
        f"Then, a series of operations (adding a chip or removing a chip) is performed. "
        f"What are the *final* chips contained in Cup {target_cup_id}? "
        f"List the chip values, separated by commas."
    )

    # --- [NEW] Generates "wrong" distractors ---

    # Interference 1: Initial state (no tracking)
    d1_initial = initial_states[target_cup_id]

    # Interference 2: Location confusion (mistaking the status of B for A)
    other_cup_id = next(cid for cid in CUP_IDS if cid != target_cup_id)
    d2_swapped = final_states[other_cup_id]

    # Distractor 3: Action confusion (add/remove reversed)
    buggy_states = {cup: chips.copy() for cup, chips in initial_states.items()}
    for op_type, cup_id, chip_id in operations:
        if op_type == 'add': # Mistaken for remove
            if chip_id in buggy_states[cup_id]:
                buggy_states[cup_id].remove(chip_id)
        elif op_type == 'remove': # Mistaken for add
            buggy_states[cup_id].append(chip_id)
    d3_action_bug = buggy_states[target_cup_id]

    # --- Generate questions and options ---
    options, correct_answer = generate_answer_options(
        final_states[target_cup_id],
        d1_initial,
        d2_swapped,
        d3_action_bug
    )

    video_filename = f"chip_cup_challenge_{challenge_id}.mp4"
    data = {
        "id": challenge_id,
        "challenge_type": "chip_cup_tracking",
        "question": question,
        "options": options,
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_filename": video_filename,
        "video_filepath": os.path.join(OUTPUT_VIDEO_DIR, video_filename),

        # The truth value passed to VideoGenerator
        "initial_states": initial_states,
        "operations": operations,
        "final_states": final_states,
        "target_cup_id_for_question": target_cup_id # [NEW] passed to metadata
    }
    return data

# --- 5. Video Generation Engine (video generator) ---
class VideoGenerator:
    ## --- MODIFICATION (Timestamp) --- ##
    # (Added self.events_timeline_ms to __init__)
    ## -------------------------------- ##
    def __init__(self, challenge_data):
        self.data = challenge_data
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.dynamic_patches = []

        # Animation constants
        self.cup_positions = { # (cx, cy)
            'A': (2.5, 3),
            'B': (5.5, 3)
        }
        self.chip_pool = {c[0]: {'value': c[1], 'color': c[2]} for c in CHIP_POOL}
        self.op_start_pos = (4, 0.5) # Where new chips appear
        self.op_end_pos = (4, 0.5)   # Remove the location where the chips go

        # --- New: Precompute "milliseconds (ms) status log" (only keep key actions) ---
        self.events_timeline_ms = []
        current_time_ms = 0.0

        # Convert from frames to milliseconds
        reveal_duration_ms = (INITIAL_REVEAL_FRAMES / FPS) * 1000
        animation_duration_ms = (OP_ANIMATION_FRAMES / FPS) * 1000
        static_duration_ms = (OP_STATIC_FRAMES / FPS) * 1000

        # 1. Record the initial display event
        self.events_timeline_ms.append({
            "event_type": "initial_reveal",
            "initial_states": self.data['initial_states'],
            "start_ms": int(current_time_ms),
            "end_ms": int(current_time_ms + reveal_duration_ms)
        })
        current_time_ms += reveal_duration_ms

        # 2. Record all operations
        for op_type, cup_id, chip_id in self.data['operations']:
            # 2a. Record the *animation* event of 'add' or 'remove'
            start_ms = int(current_time_ms)
            current_time_ms += animation_duration_ms
            end_ms = int(current_time_ms)

            self.events_timeline_ms.append({
                "event_type": f"{op_type}_chip", # "add_chip" or "remove_chip"
                "cup_id": cup_id,
                "chip_id": chip_id,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

            # 2b. Advance time (but do not log the "pause" event)
            current_time_ms += static_duration_ms

        # 3. Final pause (not recorded)
        # (self.total_frames will be calculated in generate_video)
        # --- End of new addition ---
    ## --- MODIFICATION END --- ##

    def _draw_background(self):
        """Draw desktop background"""
        self.ax.clear()
        self.ax.set_facecolor('#DFDFDF') # light gray desktop
        self.ax.set_xlim(0, 8); self.ax.set_ylim(0, 6)
        self.ax.axis('off')

    def _draw_cup(self, center_pos, label):
        """draw a cup"""
        cx, cy = center_pos
        cup = patches.Ellipse(center_pos, width=1.5, height=2.0, facecolor='#D2B48C', edgecolor='black', zorder=5) # tan
        self.dynamic_patches.append(self.ax.add_patch(cup))
        self.dynamic_patches.append(
            self.ax.text(cx, cy, label, ha='center', va='center', color='black', fontsize=20, weight='bold', zorder=10)
        )

    def _draw_chip(self, center_pos, chip_id):
        """draw a chip"""
        cx, cy = center_pos
        chip_data = self.chip_pool[chip_id]

        patch = patches.Circle(
            center_pos, radius=0.4,
            facecolor=chip_data['color'], edgecolor='black', lw=2, zorder=15
        )
        self.dynamic_patches.append(self.ax.add_patch(patch))

        # Text on the chips is in white (except white chips are in black)
        text_color = 'black' if chip_data['color'] == '#FFFFFF' else 'white'

        self.dynamic_patches.append(
            self.ax.text(cx, cy, chip_data['value'], ha='center', va='center', color=text_color, fontsize=12, weight='bold', zorder=16)
        )

    def _update(self, frame):
        # (This function remains unchanged)
        [p.remove() for p in self.dynamic_patches]; self.dynamic_patches = []
        self._draw_background()

        total_op_frames = len(self.data['operations']) * (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)
        op_start_frame = INITIAL_REVEAL_FRAMES
        op_end_frame = op_start_frame + total_op_frames

        # ---Phase 1: Initial Presentation ---
        if frame < op_start_frame:
            self.ax.text(4, 5.5, "Revealing Initial State...", ha='center', va='center', color='black', fontsize=16)
            for cup_id, pos in self.cup_positions.items():
                self._draw_cup(pos, cup_id)
                chips = self.data['initial_states'][cup_id]
                # Expand the chips below the cup
                for i, chip_id in enumerate(chips):
                    chip_pos = (pos[0] - 0.5 * (len(chips)-1) + i*1.0, pos[1] - 1.5)
                    self._draw_chip(chip_pos, chip_id)
            return

        # --- Phase 3: Final Pause (not revealed) ---
        if frame >= op_end_frame:
            self.ax.text(4, 5.5, "Operations Complete", ha='center', va='center', color='black', fontsize=16)
            for cup_id, pos in self.cup_positions.items():
                self._draw_cup(pos, cup_id)
            return

        # --- Phase 2: Exchange Process ---
        adj_frame = frame - op_start_frame
        op_idx = adj_frame // (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)
        frame_in_op = adj_frame % (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)

        if op_idx >= len(self.data['operations']): return # Protect

        # draw all cups
        for cup_id, pos in self.cup_positions.items():
            self._draw_cup(pos, cup_id)

        op_type, cup_id, chip_id = self.data['operations'][op_idx]
        target_cup_pos = self.cup_positions[cup_id]

        # 2a: animation frame
        if frame_in_op < OP_ANIMATION_FRAMES:
            progress = frame_in_op / (OP_ANIMATION_FRAMES - 1)

            if op_type == 'add':
                start_pos = np.array(self.op_start_pos)
                end_pos = np.array(target_cup_pos)
                anim_coord = start_pos + (end_pos - start_pos) * progress
                self._draw_chip(anim_coord, chip_id)

            elif op_type == 'remove':
                start_pos = np.array(target_cup_pos)
                end_pos = np.array(self.op_end_pos)
                anim_coord = start_pos + (end_pos - start_pos) * progress
                self._draw_chip(anim_coord, chip_id)

        # 2b: static frame
        else:
            # Show only cups
            pass

    def generate_video(self):
        # (This function remains unchanged)
        num_ops = len(self.data['operations'])
        total_frames = INITIAL_REVEAL_FRAMES + num_ops * (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES) + FINAL_PAUSE_FRAMES

        ani = animation.FuncAnimation(self.fig, self._update, frames=total_frames, interval=1000/FPS)

        writer = animation.writers['ffmpeg'](fps=FPS, metadata=dict(artist='Gemini'), bitrate=2000)

        try:
            ani.save(self.data['video_filepath'], writer=writer, dpi=120)
        finally:
            plt.close(self.fig) # Regardless of success or failure, close the graph to prevent memory leaks

# --- 6. Parallel Processing Worker (multi-process worker) ---
def process_single_challenge(challenge_id):
    try:
        challenge_data = generate_challenge_data(challenge_id)

        vg = VideoGenerator(challenge_data)
        vg.generate_video()

        ## --- MODIFICATION (Timestamp) --- ##
        # (Get self.events_timeline_ms status log)
        video_events_timeline_ms = vg.events_timeline_ms
        ## --- MODIFICATION END --- ##

        # --- Format SFT and RL ---
        question_text = challenge_data['question'] + "\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(challenge_data['options'])])
        video_server_path = os.path.join(VIDEO_PREFIX_IN_JSON, challenge_data['video_filename']).replace("\\", "/")

        sft = {"type":"intermedia_sft", "messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant.","loss":False}]},{"role":"user","content":[{"type":"video","uuid":str(uuid.uuid4()),"video":video_server_path},{"type":"text","text":question_text,"loss":False}]},{"role":"assistant","content":[{"type":"text","text":challenge_data['correct_answer_label'],"loss":True}]}],"extra":{"id":challenge_data['id'],"task":challenge_data['challenge_type'],"data_type":"video","data_source":"Synthetic_Video_Chip_Cup","original_video_path":video_server_path}}

        # [--- MODIFIED START ---] (exactly the same format as your previous one)
        # Create a metadata record containing both old and new formats
        metadata = {
            # RL-style format
            "problem_id": challenge_data['id'],
            "problem": f"<video>\n{question_text}",
            "data_type": "video",
            "problem_type": challenge_data['challenge_type'],
            "videos": [video_server_path],
            "data_source": "Synthetic_Video_Chip_Cup",
            "answer": challenge_data['correct_answer_label'],

            # Detailed ground truth information
            "ground_truth_details": {
                "initial_states": challenge_data['initial_states'],
                "operations": challenge_data['operations'],
                "final_states": challenge_data['final_states'],
                "question_data": {
                    "question": challenge_data['question'],
                    "target_cup_id": challenge_data['target_cup_id_for_question'],
                    "options": challenge_data['options'],
                    "correct_answer_value": challenge_data['correct_answer_value']
                },

                ## --- MODIFICATION (Timestamp) --- ##
                # (Add "milliseconds" status log here)
                "video_events_timeline_ms": video_events_timeline_ms
                ## --- MODIFICATION END --- ##
            }
        }
        # [--- MODIFIED END ---]

        ## --- FIX 2 of 2: Fix return statement --- ##
        # Error: return sft, rl, metadata
        # correct:
        return sft, metadata, metadata

    except Exception as e:
        # [--- ADDED ---] Add traceback for debugging
        return (challenge_id, f"Error in processing: {e}\n{traceback.format_exc()}")

# --- 7. Main Execution Logic (main execution logic) ---
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
             # Make sure res[2] (metadata) exists
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
