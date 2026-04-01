"""Video type: Terminal Tracking terminal command tracking video
Description: Generate a tracking video that simulates terminal command line file operations, including the following subtypes:
  - final_state_query: After all commands are executed, query which file exists
  - final_count_query: After all commands are executed, count the number of files in the directory
  - historical_query: Query whether a file has been deleted
  Simulate touch (create file) and rm (delete file) commands, character-by-character typing effect"""

import os
import json
import random
import uuid
import shutil
import concurrent.futures
import traceback  # import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm

plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

NUM_VIDEOS_TO_GENERATE = 500
NUM_WORKERS = 128 # Adjust according to your CPU core number

BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Terminal_Tracking_V2/" # [--- MODIFIED ---] It is recommended to use a new directory
OUTPUT_VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SFT_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "sft.jsonl")
RL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "rl.jsonl")
METADATA_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata.jsonl")
VIDEO_PREFIX_IN_JSON = os.path.join(BASE_OUTPUT_DIR, "videos/")

FPS = 30
TYPING_FRAMES_PER_CHAR = 2  # Typing speed per character
HOLD_FRAMES_AFTER_CMD = 45  # Pause time for "execute" command (1.5 seconds)
FINAL_PAUSE_FRAMES = 60     # final pause

# --- 2. Terminal Challenge Helpers ---

# [--- NEW ---] Define file pools and challenge types
FILE_POOL = [
    'notes.txt', 'report.pdf', 'data.csv', 'image.png', 'main.py',
    'config.json', 'README.md', 'utils.js', 'style.css', 'archive.zip',
    'test.py', 'document.docx', 'presentation.pptx', 'audio.mp3', 'video.mp4'
]

CHALLENGE_TYPES = [
    'final_state_query',    # "Which file last existed?"
    'final_count_query',    # "How many files are there at the end?"
    'historical_query'      # "Has data.csv been deleted?"
]

def generate_answer_options_files(correct_file, all_files, final_files):
    """Generate options for final_state_query"""
    options = {correct_file}

    # Interference: from the end *doesn't* exist
    decoys = set(all_files) - set(final_files)
    while len(options) < 4 and decoys:
        options.add(decoys.pop())

    # complete
    while len(options) < 4:
        options.add(f"decoy_file_{len(options)}.txt")

    options = list(options)
    random.shuffle(options)
    return options, correct_file

def generate_numeric_options(correct_answer, max_val):
    """Generate number options (infinite loop fixed)"""
    correct_answer = str(correct_answer)
    options = {correct_answer}
    possible_decoys = set(str(i) for i in range(max_val + 1)) - {correct_answer}

    while len(options) < 4 and possible_decoys:
        options.add(possible_decoys.pop())

    i = max_val + 1
    while len(options) < 4:
        options.add(str(i))
        i += 1

    options = list(options)
    random.shuffle(options)
    return options

# --- 3. Simulation Engine (core: true value simulator) ---

def simulate_operations(operations):
    """Simulates all file operations, returning final status and detailed history.
    operations = [('touch', ['a.txt']), ('rm', ['a.txt'])]"""
    current_files = set()
    history = {
        'files_created': set(),
        'files_deleted': set(),
        'files_recreated': set()
    }

    for op_type, files in operations:
        for file in files:
            if op_type == 'touch':
                if file not in current_files:
                    if file in history['files_deleted']:
                        history['files_recreated'].add(file)
                    current_files.add(file)
                    history['files_created'].add(file)

            elif op_type == 'rm':
                if file in current_files:
                    current_files.discard(file)
                    history['files_deleted'].add(file)

    return current_files, history

# --- 4. Challenge Generation Logic (Core: Challenge Generator) ---
# [--- No changes ---]
def generate_challenge_data(challenge_id):

    num_ops = random.randint(5, 8) # 5-8 commands
    operations = []

    # --- Intelligent generation operation ---
    sim_files = set()
    available_files = set(random.sample(FILE_POOL, 10)) # Files available in this video

    for _ in range(num_ops):
        op_type = random.choice(['touch', 'rm'])

        # Decide how many files to operate (1-3)
        num_files_in_op = random.randint(1, 3)
        files_to_op = []

        if op_type == 'touch':
            # 50% chance of creating a new file, 50% chance of creating an existing file (harmless)
            for _ in range(num_files_in_op):
                if random.random() < 0.5 and available_files:
                    files_to_op.append(available_files.pop())
                elif sim_files:
                    files_to_op.append(random.choice(list(sim_files)))
                elif available_files: # spare
                    files_to_op.append(available_files.pop())
            if not files_to_op: continue # No more files

        elif op_type == 'rm':
            # 50% probability of deleting existing files, 50% probability of deleting non-existent files (harmless)
            for _ in range(num_files_in_op):
                if random.random() < 0.5 and sim_files:
                    files_to_op.append(random.choice(list(sim_files)))
                elif available_files:
                    files_to_op.append(available_files.pop()) # Try deleting a file that was never created
            if not files_to_op: continue # No more files

        # Update simulation status
        for f in files_to_op:
            if op_type == 'touch':
                sim_files.add(f)
            elif op_type == 'rm':
                sim_files.discard(f)

        operations.append((op_type, files_to_op))

    # --- Run a true simulation ---
    final_files, history = simulate_operations(operations)

    # --- Randomly select a challenge type ---
    challenge_type = random.choice(CHALLENGE_TYPES)

    # If the history is empty (e.g. rm rms one that does not exist), downgrade
    if not history['files_deleted'] and challenge_type == 'historical_query':
        challenge_type = 'final_count_query'

    if not final_files and challenge_type == 'final_state_query':
        challenge_type = 'final_count_query'

    question = ""
    correct_answer = ""
    options = []

    if challenge_type == 'final_state_query':
        correct_file = random.choice(list(final_files))
        question = "After all commands are executed, which of the following files exists in the directory?"
        options, correct_answer = generate_answer_options_files(correct_file, FILE_POOL, final_files)

    elif challenge_type == 'final_count_query':
        correct_count = len(final_files)
        question = "After all commands are executed, how many files are in the directory?"
        options = generate_numeric_options(correct_count, max_val=len(FILE_POOL))
        correct_answer = str(correct_count)

    elif challenge_type == 'historical_query':
        # Randomly select a deleted file
        target_file = random.choice(list(history['files_deleted']))
        question = f"During the process, was the file '{target_file}' ever deleted?"

        # The answer depends on whether it was recreated
        if target_file in history['files_recreated']:
             # For example: touch a -> rm a -> touch a
             # "Has a been deleted?" -> Yes
             correct_answer = "Yes"
        else:
             # For example: touch a -> rm a
             correct_answer = "Yes"

        # [--- NEW ---] Intelligent distractor issues
        if random.random() < 0.3:
            # Switch to "never existed" question
            target_file = (available_files.pop() if available_files else "non_existent.file")
            question = f"During the process, was the file '{target_file}' ever deleted?"
            correct_answer = "No"

        options = ["Yes", "No"]


    video_filename = f"terminal_challenge_{challenge_id}.mp4"
    data = {
        "id": challenge_id,
        "challenge_type": challenge_type,
        "question": question,
        "options": options,
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_filename": video_filename,
        "video_filepath": os.path.join(OUTPUT_VIDEO_DIR, video_filename),

        # The truth value passed to VideoGenerator
        "operations": operations,
        "final_files": final_files,
        "history": history
    }
    return data

# --- 5. Video Generation Engine ---
class VideoGenerator:
    ## --- MODIFICATION (Timestamp) --- ##
    # (Added self.events_timeline_ms in __init__)
    ## -------------------------------- ##
    def __init__(self, challenge_data):
        self.data = challenge_data

        self.fig = plt.figure(figsize=(8, 6))
        self.fig.set_facecolor('black')

        self.ax = self.fig.add_axes([0, 0, 1, 1])

        self.dynamic_patches = []

        # Animation constants
        self.prompt = "$ "

        # 1. "Flat" operations into frame events
        self.frames_events = []

        ## --- New: Precompute "milliseconds (ms) status log" (only keep typing_command) --- ##
        self.events_timeline_ms = []
        current_time_ms = 0.0

        # Convert from frames to milliseconds
        typing_char_duration_ms = (TYPING_FRAMES_PER_CHAR / FPS) * 1000
        hold_cmd_duration_ms = (HOLD_FRAMES_AFTER_CMD / FPS) * 1000
        clear_duration_ms = ((HOLD_FRAMES_AFTER_CMD // 3) / FPS) * 1000
        ## --- End of new addition --- ##

        for op_type, files in self.data['operations']:
            cmd_string = f"{op_type} {' '.join(files)}"

            ## --- MODIFICATION (Timestamp) --- ##
            # 1. Calculate the time of typing event
            typing_duration_ms = len(cmd_string) * typing_char_duration_ms

            start_ms = int(current_time_ms)
            current_time_ms += typing_duration_ms
            end_ms = int(current_time_ms)

            # --- Add only typing_command event ---
            self.events_timeline_ms.append({
                "event_type": "typing_command",
                "command_string": cmd_string,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

            # 2. Advance time (but do not log "execute" and "clear" events)
            current_time_ms += hold_cmd_duration_ms
            current_time_ms += clear_duration_ms
            ## --- MODIFICATION END --- ##

            # (The list of frame events below remains unchanged and is only used to generate the video)
            # 2. Typing
            for i in range(1, len(cmd_string) + 1):
                self.frames_events.extend([
                    {'state': 'typing', 'text': self.prompt + cmd_string[:i]}
                ] * TYPING_FRAMES_PER_CHAR)

            # 3. keep
            self.frames_events.extend([
                {'state': 'holding', 'text': self.prompt + cmd_string}
            ] * HOLD_FRAMES_AFTER_CMD)

            # 4. Clear (show empty prompt)
            self.frames_events.extend([
                {'state': 'prompt', 'text': self.prompt}
            ] * (HOLD_FRAMES_AFTER_CMD // 3)) # A short pause after "pressing enter"

        # 4. Final pause
        self.frames_events.extend([
            {'state': 'prompt', 'text': self.prompt}
        ] * FINAL_PAUSE_FRAMES)

        self.total_frames = len(self.frames_events)


    def _draw_background(self):
        """Draw black terminal background"""
        self.ax.clear()

        self.ax.set_facecolor('black')

        self.ax.set_xlim(0, 10); self.ax.set_ylim(0, 10)
        self.ax.axis('off')

    def _update(self, frame):
        # (This function remains unchanged)
        [p.remove() for p in self.dynamic_patches]; self.dynamic_patches = []
        self._draw_background()

        if frame >= self.total_frames:
            event = self.frames_events[-1] # stay on last frame
        else:
            event = self.frames_events[frame]

        # draw text
        self.dynamic_patches.append(
            self.ax.text(0.5, 9, event['text'], ha='left', va='top', color='white', fontsize=14, fontfamily='monospace')
        )

        # Draw a blinking cursor
        if event['state'] != 'holding' and (frame % (FPS//2) < (FPS//4)):
             # Calculate cursor position
             text_width_estimate = len(event['text']) * 0.35 # Estimate
             self.dynamic_patches.append(
                 self.ax.text(0.5 + text_width_estimate, 9, "\u258b", ha='left', va='top', color='white', fontsize=14, fontfamily='monospace')
             )

    def generate_video(self):
        # (This function remains unchanged)
        ani = animation.FuncAnimation(self.fig, self._update, frames=self.total_frames, interval=1000/FPS)

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

        # --- Formatting SFT and RL ---
        question_text = challenge_data['question'] + "\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(challenge_data['options'])])
        video_server_path = os.path.join(VIDEO_PREFIX_IN_JSON, challenge_data['video_filename']).replace("\\", "/")

        sft = {"type":"intermedia_sft", "messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant.","loss":False}]},{"role":"user","content":[{"type":"video","uuid":str(uuid.uuid4()),"video":video_server_path},{"type":"text","text":question_text,"loss":False}]},{"role":"assistant","content":[{"type":"text","text":challenge_data['correct_answer_label'],"loss":True}]}],"extra":{"id":challenge_data['id'],"task":challenge_data['challenge_type'],"data_type":"video","data_source":"Synthetic_Video_Terminal","original_video_path":video_server_path}}

        # [--- MODIFIED ---]
        # To unify with Trajectory_Math_v2, we merge the RL and Metadata formats
        # RL records will now contain ground_truth_details

        rl_and_metadata = {
            # RL-style format (your original rl dictionary)
            "problem_id": challenge_data['id'],
            "problem": f"<video>\n{question_text}",
            "data_type": "video",
            "problem_type": challenge_data['challenge_type'],
            "videos": [video_server_path],
            "data_source": "Synthetic_Video_Terminal",
            "answer": challenge_data['correct_answer_label'],

            # Detailed ground truth information (your original metadata dictionary)
            "ground_truth_details": {
                "operations": challenge_data['operations'],
                "final_files": list(challenge_data['final_files']), # set -> list
                "history": {
                    'files_created': list(challenge_data['history']['files_created']),
                    'files_deleted': list(challenge_data['history']['files_deleted']),
                    'files_recreated': list(challenge_data['history']['files_recreated'])
                },
                "question_data": {
                    "question": challenge_data['question'],
                    "options": challenge_data['options'],
                    "correct_answer_value": challenge_data['correct_answer_value']
                },

                ## --- MODIFICATION (Timestamp) --- ##
                # (Add "milliseconds" status log here)
                "video_events_timeline_ms": video_events_timeline_ms
                ## --- MODIFICATION END --- ##
            }
        }

        # (We now return sft, rl_and_metadata, rl_and_metadata)
        # (This way the contents of rl.jsonl and metadata.jsonl will be consistent)
        return sft, rl_and_metadata, rl_and_metadata

    except Exception as e:
        # Add traceback printing
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
