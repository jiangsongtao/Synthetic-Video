"""Video type: Grid Movement Tracking grid movement tracking video
Description: Generate tracking video of a small ball performing long-distance random/pattern walking on an 8x8 grid, including the following subtypes:
  - furthest_from_origin: identifies the access coordinate farthest from the origin
  - visited_position: Determine whether the ball has passed a specific coordinate
  - direction_count: counts the number of times the ball moves in a specific direction
  - pattern_return: predict the next direction of a structured repeating pattern"""

import cv2
import numpy as np
import os
import random
import json
import uuid
import imageio
from collections import Counter

# --- Configuration ---
BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/Longtern_synthetic"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ==============================================================================
#  VIDEO GENERATION HELPERS
# ==============================================================================

def create_video_writer(output_path, fps):
    return imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None, quality=8)

def append_frame(writer, frame):
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def draw_grid_with_labels(frame, grid_size, cell_size, start_x, start_y):
    for i in range(grid_size + 1):
        cv2.line(frame, (start_x, start_y + i * cell_size), (start_x + grid_size * cell_size, start_y + i * cell_size), (200, 200, 200), 2)
        cv2.line(frame, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size * cell_size), (200, 200, 200), 2)
        if i < grid_size:
            cv2.putText(frame, str(i), (start_x - 30, start_y + i * cell_size + cell_size//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, str(i), (start_x + i * cell_size + cell_size//2 - 10, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return frame

def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# ==============================================================================
#  COMPLEX MOVEMENT TRACKING VIDEO GENERATOR
# ==============================================================================

def create_complex_movement_tracking_video(output_path, fps=2, target_duration=30, question_type="random"):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)

    grid_size, cell_size, start_x, start_y = 8, 60, 150, 50 # Larger grid
    num_steps = int(target_duration * fps)

    path = [(grid_size // 2, grid_size // 2)] # Start near center
    origin = (0, 0)
    directions = []
    direction_map = {
        (0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"
    }

    metadata = {}

    # --- Task-Specific Path and Question Generation ---
    if question_type == 'pattern_return':
        # REDESIGNED LOGIC
        outbound_pattern = random.choice([[(0, 1), (0, 1), (-1, 0)], [(1, 0), (0, -1), (0, -1)], [(-1, 0), (0, 1)]])
        return_pattern = [(-dr, -dc) for dr, dc in reversed(outbound_pattern)]

        repeats = 3 # How many times to repeat the outbound/return journey

        # Construct the full theoretical path
        full_move_sequence = (outbound_pattern * repeats) + (return_pattern * repeats)

        # Truncate the path to fit the video duration, but ensure it's interrupted
        interrupt_point = min(num_steps - 1, len(full_move_sequence) - random.randint(1, 2))
        video_moves = full_move_sequence[:interrupt_point]

        # The correct next move is the one that was cut off
        next_move_vector = full_move_sequence[interrupt_point]
        correct_answer = direction_map.get(next_move_vector, "Unknown")

        # Generate the actual path on the grid
        for dr, dc in video_moves:
            r, c = path[-1]
            nr, nc = r + dr, c + dc
            # Ensure the path stays within bounds by wrapping around (toroidal grid)
            nr = nr % grid_size
            nc = nc % grid_size
            path.append((nr, nc))
            directions.append(direction_map.get((dr, dc), "Unknown"))

        metadata = {
            # IMPLICIT QUESTION
            "question": "The ball's movement appears to follow a structured, repeating pattern. After the video ends, what is the most likely next direction of movement?",
            "answer": correct_answer,
            "distractors": [d for d in direction_map.values() if d != correct_answer]
        }

    else:
        # Biased random walk for other tasks
        visit_counts = Counter({path[0]: 1})
        for _ in range(num_steps - 1):
            r, c = path[-1]
            possible_moves = []
            for dr, dc in direction_map.keys():
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    weight = 1.0 / (visit_counts.get((nr, nc), 0) + 1)
                    possible_moves.append(((dr, dc), weight))

            if not possible_moves: path.append((r,c)); directions.append("Stay"); continue

            total_weight = sum(w for _, w in possible_moves)
            rand_val = random.uniform(0, total_weight)
            cumulative_weight = 0
            chosen_move = possible_moves[-1][0]
            for move, weight in possible_moves:
                cumulative_weight += weight
                if rand_val <= cumulative_weight: chosen_move = move; break

            dr, dc = chosen_move
            nr, nc = r + dr, c + dc
            path.append((nr, nc))
            directions.append(direction_map.get((dr, dc)))
            visit_counts[(nr, nc)] += 1

        # Calculate Answers for the generated path
        distances_from_origin = sorted([(pos, calculate_distance(pos, origin)) for pos in path], key=lambda x: x[1], reverse=True)
        furthest_pos = distances_from_origin[0][0]

        if question_type == "furthest_from_origin":
            metadata = {
                "question": "Considering the entire journey, which coordinate (row, col) visited by the ball was furthest from the origin (0, 0)?",
                "answer": f"({furthest_pos[0]}, {furthest_pos[1]})",
                "distractors": [f"({p[0][0]}, {p[0][1]})" for p in distances_from_origin[1:5]]
            }

        elif question_type == "visited_position":
            path_visit_counts = Counter(path)
            once_visited_points = [p for p, c in path_visit_counts.items() if c == 1]

            if random.random() < 0.6 and once_visited_points:
                target_point = random.choice(once_visited_points)
                correct = "Yes"
            else:
                all_points = set((r, c) for r in range(grid_size) for c in range(grid_size))
                unvisited = list(all_points - set(path))
                target_point = random.choice(unvisited) if unvisited else (0,0)
                correct = "No"

            metadata = {
                "question": f"Throughout its entire path, did the ball land on the coordinate ({target_point[0]}, {target_point[1]})?",
                "answer": correct, "distractors": ["Yes", "No"]
            }

        elif question_type == "direction_count":
            direction_counter = Counter(directions)
            valid_directions = [d for d, c in direction_counter.items() if c > 0 and d != "Stay"]
            target_direction = random.choice(valid_directions) if valid_directions else "Right"
            count = direction_counter.get(target_direction, 0)
            metadata = {
                "question": f"Over its entire path, how many times did the ball move in the '{target_direction}' direction?",
                "answer": str(count),
                "distractors": [str(c) for c in [count-2, count+1, count+3, 0] if c >= 0 and c != count]
            }

    # --- GENERATE VIDEO FRAMES (PURE VISUALS ONLY) ---
    base_frame = draw_grid_with_labels(np.ones((height, width, 3), dtype=np.uint8) * 255, grid_size, cell_size, start_x, start_y)

    for r, c in path:
        frame = base_frame.copy()
        center_x, center_y = start_x + c * cell_size + cell_size // 2, start_y + r * cell_size + cell_size // 2
        cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
        append_frame(writer, frame)

    writer.close()

    # --- FINAL METADATA ---
    duration = len(path) / fps
    print(f"✅ Video saved: {output_path}")
    print(f"📊 Question Type: {question_type}")
    print(f"📊 Answer: {metadata['answer']}")

    metadata.update({"width": width, "height": height, "fps": fps, "duration": duration, "task_type": f"Advanced Movement: {question_type}"})
    return metadata

# ==============================================================================
#  MAIN EXECUTION SCRIPT
# ==============================================================================
def create_all_movement_tracking_videos(num_videos_per_type=2):
    # Replaced the old pattern prediction with the new, more robust pattern_return
    question_types = ["furthest_from_origin", "visited_position", "direction_count", "pattern_return"]
    sft_data, rl_data, all_metadata = [], [], []
    video_id = 1

    for q_type in question_types:
        for i in range(num_videos_per_type):
            target_duration = random.randint(25, 40)
            filename = f"movement_tracking_{q_type}_{i+1}.mp4"
            video_path = os.path.join(BASE_OUTPUT_DIR, filename)

            print(f"\n{'='*70}\n📹 Generating video {video_id}: {filename}\n   Question Type: {q_type}, Target Duration: {target_duration}s\n{'='*70}")

            try:
                metadata = create_complex_movement_tracking_video(output_path=video_path, fps=2, target_duration=target_duration, question_type=q_type)
                all_metadata.append(metadata)

                options_text, correct_letter = create_multiple_choice_data(metadata["question"], metadata["answer"], metadata["distractors"])
                full_question_text = f"{metadata['question']}\n{options_text}"

                sft_entry = {"type": "intermedia_sft", "messages": [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant.", "loss": False}]}, {"role": "user", "content": [{"type": "video", "uuid": str(uuid.uuid4()), "video": video_path, "width": metadata["width"], "height": metadata["height"], "duration": metadata["duration"], "fps": metadata["fps"], "timestamp_fmt": "seconds", "has_bisec_tar": False}, {"type": "text", "text": full_question_text, "loss": False}]}, {"role": "assistant", "content": [{"type": "text", "text": correct_letter, "loss": True}]}], "extra": {"extra": {"id": video_id, "doc_id": video_id, "task": metadata["task_type"], "data_type": "video", "data_source": "Synthetic_Movement_Tracking", "original_video_path": video_path}}}
                sft_data.append(sft_entry)

                rl_entry = {"problem_id": video_id, "problem": f"<video>{full_question_text}", "data_type": "video", "problem_type": metadata["task_type"], "videos": [video_path], "data_source": "Synthetic_Movement_Tracking", "answer": correct_letter}
                rl_data.append(rl_entry)

                video_id += 1
            except Exception as e:
                print(f"❌ Error generating {filename}: {e}")
                import traceback
                traceback.print_exc()

    return sft_data, rl_data, all_metadata

def create_multiple_choice_data(question, correct_answer, distractors):
    if correct_answer in distractors: options = list(set(distractors + [correct_answer]))
    else:
        available_distractors = [d for d in distractors if d != correct_answer]
        num_distractors = min(3, len(available_distractors))
        options = [correct_answer] + random.sample(available_distractors, num_distractors) if available_distractors else [correct_answer]

    random.shuffle(options)
    correct_letter = chr(ord('A') + options.index(correct_answer))
    options_text = "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)])
    return options_text, correct_letter

def write_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data_list: f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✅ Successfully wrote {len(data_list)} entries to {filename}")

def write_metadata_json(metadata_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Successfully wrote metadata to {filename}")

def main():
    print("\n" + "="*70 + "\n🎬 ADVANCED MOVEMENT TRACKING VIDEO GENERATOR\n" + "="*70)

    sft_data, rl_data, all_metadata = create_all_movement_tracking_videos(num_videos_per_type=2)

    if not all_metadata:
        print("\n❌ No videos were generated. Exiting.")
        return

    print("\n" + "="*70 + "\n💾 Saving data files...\n" + "="*70)
    sft_output_path = os.path.join(BASE_OUTPUT_DIR, "movement_tracking_sft_data.jsonl")
    rl_output_path = os.path.join(BASE_OUTPUT_DIR, "movement_tracking_rl_data.jsonl")
    metadata_output_path = os.path.join(BASE_OUTPUT_DIR, "movement_tracking_metadata.json")
    write_jsonl(sft_data, sft_output_path)
    write_jsonl(rl_data, rl_output_path)
    write_metadata_json(all_metadata, metadata_output_path)

    print("\n" + "="*70 + "\n📊 GENERATION SUMMARY\n" + "="*70)
    print(f"✨ Total videos generated: {len(all_metadata)}")

    task_types = Counter(meta.get("task_type", "Unknown") for meta in all_metadata)
    print("\n📊 Task Type Distribution:")
    for task, count in task_types.items(): print(f"   - {task}: {count} videos")

    total_duration = sum(meta.get("duration", 0) for meta in all_metadata)
    print(f"\n⏱️  Total video duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    if all_metadata: print(f"⏱️  Average video duration: {total_duration/len(all_metadata):.2f} seconds")

    print("\n" + "="*70 + "\n✅ ALL DONE!\n" + "="*70)

if __name__ == "__main__":
    main()
