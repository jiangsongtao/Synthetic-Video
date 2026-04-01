"""Run all video generation scripts to generate 1 example video per type
Output to the /cpfs01/user/songtao.jst/video_scripts/examples/ directory"""
import subprocess
import sys
import os

EXAMPLES_DIR = "/cpfs01/user/songtao.jst/video_scripts/examples"
SCRIPTS_DIR = "/cpfs01/user/songtao.jst/video_scripts"

# Each entry: (script_name, list of (old_str, new_str) replacements)
PATCHES = {
    "01_atomic_motion.py": [
        ('"output_dir": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/"',
         f'"output_dir": "{EXAMPLES_DIR}/01_atomic_motion/"'),
        ('"sft_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/sft.jsonl"',
         f'"sft_filename": "{EXAMPLES_DIR}/01_atomic_motion/sft.jsonl"'),
        ('"metadata_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic/metadata.jsonl"',
         f'"metadata_filename": "{EXAMPLES_DIR}/01_atomic_motion/metadata.jsonl"'),
        ('generate_dataset(samples_per_type=500, max_workers=128)',
         'generate_dataset(samples_per_type=1, max_workers=1)'),
    ],
    "02_atomic2_extended_motion.py": [
        ('"output_dir": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/"',
         f'"output_dir": "{EXAMPLES_DIR}/02_atomic2_extended_motion/"'),
        ('"sft_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/sft.jsonl"',
         f'"sft_filename": "{EXAMPLES_DIR}/02_atomic2_extended_motion/sft.jsonl"'),
        ('"metadata_filename": "/cpfs02/user/songtao.jst/reasoning_videos/RL/Atomic2/metadata.jsonl"',
         f'"metadata_filename": "{EXAMPLES_DIR}/02_atomic2_extended_motion/metadata.jsonl"'),
        ('generate_dataset(samples_per_type=200, max_workers=128)',
         'generate_dataset(samples_per_type=1, max_workers=1)'),
    ],
    "03_shell_game.py": [
        ('NUM_PER_TYPE = 60', 'NUM_PER_TYPE = 1'),
        ('NUM_WORKERS = 64', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/ball_swicth_V2/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/03_shell_game/"'),
    ],
    "04_sliding_puzzle.py": [
        ('NUM_PER_TYPE = 150', 'NUM_PER_TYPE = 1'),
        ('NUM_WORKERS = 64', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Sliding_Puzzle_v2_4x4_V2/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/04_sliding_puzzle/"'),
    ],
    "05_card_pile.py": [
        ('NUM_VIDEOS_TO_GENERATE = 500', 'NUM_VIDEOS_TO_GENERATE = 1'),
        ('NUM_WORKERS = 64', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Card_Pile_V2/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/05_card_pile/"'),
    ],
    "06_chip_cup.py": [
        ('NUM_VIDEOS_TO_GENERATE = 500', 'NUM_VIDEOS_TO_GENERATE = 1'),
        ('NUM_WORKERS = 64', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Chip_Cup_v2_EN/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/06_chip_cup/"'),
    ],
    "07_terminal_tracking.py": [
        ('NUM_VIDEOS_TO_GENERATE = 500', 'NUM_VIDEOS_TO_GENERATE = 1'),
        ('NUM_WORKERS = 128', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Terminal_Tracking_V2/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/07_terminal_tracking/"'),
    ],
    "08_trajectory_math.py": [
        ('NUM_VIDEOS_TO_GENERATE = 500', 'NUM_VIDEOS_TO_GENERATE = 1'),
        ('NUM_WORKERS = 64', 'NUM_WORKERS = 1'),
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Trajectory_Math_v2/"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/08_trajectory_math/"'),
    ],
    "09_longterm_temporal.py": [
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/Longtern_synthetic"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/09_longterm_temporal"'),
    ],
    "10_grid_movement_tracking.py": [
        ('BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/Longtern_synthetic"',
         f'BASE_OUTPUT_DIR = "{EXAMPLES_DIR}/10_grid_movement_tracking"'),
        ('num_videos_per_type=2', 'num_videos_per_type=1'),
    ],
}


def run_one(script_name):
    src_path = os.path.join(SCRIPTS_DIR, script_name)
    tmp_path = os.path.join(SCRIPTS_DIR, f"_tmp_{script_name}")

    with open(src_path, 'r') as f:
        code = f.read()

    for old, new in PATCHES[script_name]:
        if old not in code:
            print(f"  [WARN] Patch target not found in {script_name}: {old[:60]}...")
        code = code.replace(old, new)

    with open(tmp_path, 'w') as f:
        f.write(code)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=900,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  [OK] {script_name} succeeded")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"    {line}")
        else:
            print(f"  [FAIL] {script_name} returned code {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-10:]:
                    print(f"    {line}")
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {script_name} timed out after 600s")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    scripts = sorted(PATCHES.keys())

    # If a script name is provided as arg, run only that one
    if len(sys.argv) > 1:
        scripts = [s for s in sys.argv[1:] if s in PATCHES]

    for script_name in scripts:
        run_one(script_name)

    print(f"\n{'='*60}")
    print(f"All done! Examples are in: {EXAMPLES_DIR}")
    print(f"{'='*60}")
