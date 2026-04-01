"""
Video type: Long-term Temporal Reasoning long-term temporal reasoning video
Description: Generate synthetic videos that require long-term temporal reasoning capabilities, including the following subtypes:
- path_tracing_recall: The red ball follows a random path on the 5x5 grid, recalling the last step direction
- sequential_pattern_completion: Repeating pattern sequence of colored shapes, predicting next shape
- cumulative_count: Cumulative comparison of the frequency of occurrence of red circles and blue squares
- state_tracking_occlusion: Tracking the number of chips under occlusion
- backward_inference: Reverse reasoning about the initial state based on the final state and operations
"""

import cv2
import numpy as np
import os
import random
import json
import uuid
import imageio

# --- Configuration ---
BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/Longtern_synthetic"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
MIN_VIDEO_DURATION_S = 20

# ==============================================================================
#  VIDEO GENERATION HELPERS
# ==============================================================================

def create_video_writer(output_path, fps):
    return imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None, quality=8)

def append_frame(writer, frame):
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def draw_grid_with_labels(frame, grid_size, cell_size, start_x, start_y):
    """Helper to draw a labeled grid on a frame."""
    for i in range(grid_size + 1):
        cv2.line(frame, (start_x, start_y + i * cell_size), (start_x + grid_size * cell_size, start_y + i * cell_size), (200, 200, 200), 2)
        cv2.line(frame, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size * cell_size), (200, 200, 200), 2)
        if i < grid_size:
            cv2.putText(frame, str(i), (start_x - 30, start_y + i * cell_size + cell_size//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(frame, str(i), (start_x + i * cell_size + cell_size//2 - 10, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return frame

# ==============================================================================
#  NEW, ADVANCED TEMPORAL REASONING TASKS
# ==============================================================================

def create_path_tracing_recall_video(output_path, fps=2):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)
    grid_size, cell_size, start_x, start_y = 5, 80, 200, 100

    path = [(grid_size//2, grid_size//2)]
    visited = {path[0]}
    num_steps = int(MIN_VIDEO_DURATION_S * fps)
    for _ in range(num_steps - 1):
        r, c = path[-1]
        possible_moves = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in visited:
                possible_moves.append((nr, nc))
        if not possible_moves: break
        next_pos = random.choice(possible_moves)
        path.append(next_pos)
        visited.add(next_pos)

    last_pos, second_last_pos = path[-1], path[-2]
    dr, dc = last_pos[0] - second_last_pos[0], last_pos[1] - second_last_pos[1]
    direction_map = {(0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"}
    correct_answer = direction_map.get((dr, dc), "Unknown")

    base_frame = draw_grid_with_labels(np.ones((height, width, 3), dtype=np.uint8) * 255, grid_size, cell_size, start_x, start_y)

    for i in range(len(path)):
        frame = base_frame.copy()
        for j in range(i):
            p1 = (start_x + path[j][1] * cell_size + cell_size//2, start_y + path[j][0] * cell_size + cell_size//2)
            p2 = (start_x + path[j+1][1] * cell_size + cell_size//2, start_y + path[j+1][0] * cell_size + cell_size//2)
            cv2.line(frame, p1, p2, (255, 150, 150), 3)
        r, c = path[i]
        center_x, center_y = start_x + c * cell_size + cell_size // 2, start_y + r * cell_size + cell_size // 2
        cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
        append_frame(writer, frame)

    writer.close()
    print(f"✅ Video saved: {output_path}\n📊 Answer: {correct_answer}")
    return {"width": width, "height": height, "fps": fps, "question": "A ball traced a path on the grid. What was the direction of its very last move?", "answer": correct_answer, "distractors": [d for d in direction_map.values() if d != correct_answer], "task_type": "Path Tracing Final Move"}

def create_sequential_pattern_completion_video(output_path, fps=2):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)
    shapes_map = {
        "Red Circle": lambda fr: cv2.circle(fr, (400, 300), 100, (0, 0, 255), -1),
        "Green Square": lambda fr: cv2.rectangle(fr, (300, 200), (500, 400), (0, 255, 0), -1),
        "Blue Triangle": lambda fr: cv2.fillPoly(fr, [np.array([[400, 200], [300, 400], [500, 400]], np.int32)], (255, 0, 0))
    }
    pattern = ["Red Circle", "Green Square", "Blue Triangle"]
    num_repeats = (MIN_VIDEO_DURATION_S // (len(pattern) * 2)) + 1
    sequence = pattern * num_repeats
    video_sequence, correct_answer = sequence[:-1], sequence[-1]

    blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for shape_name in video_sequence:
        frame = blank_frame.copy()
        shapes_map[shape_name](frame)
        for _ in range(fps * 2): append_frame(writer, frame)

    writer.close()
    print(f"✅ Video saved: {output_path}\n📊 Answer: {correct_answer}")
    return {"width": width, "height": height, "fps": fps, "question": "A sequence of colored shapes was shown, which followed a repeating pattern. What should the next shape in the sequence have been?", "answer": correct_answer, "distractors": [s for s in shapes_map.keys() if s != correct_answer], "task_type": "Sequential Pattern Completion"}

def create_cumulative_count_video(output_path, fps=4):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)
    num_events = int(MIN_VIDEO_DURATION_S * 1.5)
    sequence = [random.choice(["Red Circle", "Blue Square"]) for _ in range(num_events)]

    count_red = sequence.count("Red Circle")
    count_blue = sequence.count("Blue Square")
    if count_red > count_blue: correct_answer = "More Red Circles"
    elif count_blue > count_red: correct_answer = "More Blue Squares"
    else: correct_answer = "An equal number"

    for item in sequence:
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        x, y = random.randint(100, width-100), random.randint(100, height-100)
        if item == "Red Circle": cv2.circle(frame, (x, y), 40, (0, 0, 255), -1)
        else: cv2.rectangle(frame, (x-40, y-40), (x+40, y+40), (255, 0, 0), -1)
        for _ in range(int(fps * 0.6)): append_frame(writer, frame)

    writer.close()
    print(f"✅ Video saved: {output_path}\n📊 Answer: {correct_answer}")
    # FIX: Removed the trailing comma that caused the tuple error.
    return {"width": width, "height": height, "fps": fps, "question": "Red circles and blue squares appeared on the screen. After watching the entire sequence, which of the following is true?", "answer": correct_answer, "distractors": ["More Red Circles", "More Blue Squares", "An equal number"], "task_type": "Cumulative Count Comparison"}

def create_state_tracking_occlusion_video(output_path, fps=2):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)
    initial_chips = [5, 8, 3]
    cup_pos = [(200, 300), (400, 300), (600, 300)]
    cup_color = (160, 160, 160)

    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for pos, count in zip(cup_pos, initial_chips):
        cv2.rectangle(frame, (pos[0]-50, pos[1]-50), (pos[0]+50, pos[1]+50), cup_color, -1)
        cv2.putText(frame, str(count), (pos[0]-20, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
    for _ in range(fps * 4): append_frame(writer, frame)

    covered_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for pos in cup_pos: cv2.rectangle(covered_frame, (pos[0]-50, pos[1]-50), (pos[0]+50, pos[1]+50), (50, 50, 50), -1)
    for _ in range(fps * 2): append_frame(writer, covered_frame)

    actions = [(0, "+1"), (2, "-1"), (1, "+1"), (0, "-1"), (2, "+1")]
    final_chips = list(initial_chips)
    for cup_idx, op_text in actions:
        action_frame = covered_frame.copy()
        pos, color = cup_pos[cup_idx], (0, 255, 0) if "+" in op_text else (0, 0, 255)
        cv2.putText(action_frame, op_text, (pos[0]-35, pos[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        for _ in range(fps * 2): append_frame(writer, action_frame)
        final_chips[cup_idx] += 1 if "+" in op_text else -1
        for _ in range(fps * 1): append_frame(writer, covered_frame)

    writer.close()
    question_cup_idx = 1
    correct_answer = final_chips[question_cup_idx]
    print(f"✅ Video saved: {output_path}\n📊 Answer: {correct_answer}")
    # FIX: Removed the trailing comma that caused the tuple error.
    return {"width": width, "height": height, "fps": fps, "question": f"The initial chip counts were shown, then the cups were covered. After a series of operations on the covered cups, what is the final number of chips in the second cup?", "answer": str(correct_answer), "distractors": [str(initial_chips[question_cup_idx]), str(correct_answer+1), str(correct_answer-1)], "task_type": "Occluded State Tracking"}

def create_backward_inference_video(output_path, fps=2):
    width, height = 800, 600
    writer = create_video_writer(output_path, fps)
    initial_chips, final_chips = [5, 8, 3], [4, 9, 3]
    cup_pos = [(200, 300), (400, 300), (600, 300)]

    covered_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for pos in cup_pos: cv2.rectangle(covered_frame, (pos[0]-50, pos[1]-50), (pos[0]+50, pos[1]+50), (160,160,160), -1)

    for i in range(20):
        frame = covered_frame.copy()
        x, y = cup_pos[0][0] + i * 10, cup_pos[0][1] - 80 - int(20*np.sin(i/20*np.pi))
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        append_frame(writer, frame)

    for _ in range(fps * 15): append_frame(writer, covered_frame)

    final_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for pos, count in zip(cup_pos, final_chips):
        cv2.rectangle(final_frame, (pos[0]-50, pos[1]-50), (pos[0]+50, pos[1]+50), (160,160,160), -1)
        cv2.putText(final_frame, str(count), (pos[0]-20, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
    for _ in range(fps * 5): append_frame(writer, final_frame)

    writer.close()
    correct_answer = initial_chips[0]
    print(f"✅ Video saved: {output_path}\n📊 Answer: {correct_answer}")
    # FIX: Added the missing 'task_type' key.
    return {"width": width, "height": height, "fps": fps, "question": "The video showed a chip being moved between two covered cups, followed by a long pause, then it revealed the final number of chips. Based on the action and the final state, how many chips did the first cup have initially?", "answer": str(correct_answer), "distractors": [str(final_chips[0]), str(initial_chips[1]), str(final_chips[1])], "task_type": "Backward State Inference"}


# ==============================================================================
#  MAIN SCRIPT
# ==============================================================================
def create_multiple_choice_data(question, correct_answer, distractors):
    if correct_answer in distractors:
        options = distractors
    else:
        options = [correct_answer] + random.sample(distractors, min(3, len(distractors)))
    random.shuffle(options)
    correct_letter = chr(ord('A') + options.index(correct_answer))
    options_text = "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)])
    return options_text, correct_letter

def get_video_duration(video_path):
    try:
        with imageio.get_reader(video_path) as reader: return reader.get_meta_data().get('duration', 0)
    except Exception: return 0

def write_jsonl(data_list, filename):
    with open(filename, 'w') as f:
        for entry in data_list: f.write(json.dumps(entry) + '\n')
    print(f"Successfully wrote {len(data_list)} entries to {filename}")

def generate_all_videos_and_metadata():
    print("🎬 Starting generation of ADVANCED temporal reasoning videos...\n")

    videos_to_generate = [
        ("path_tracing_recall.mp4", create_path_tracing_recall_video),
        ("sequential_pattern_completion.mp4", create_sequential_pattern_completion_video),
        ("cumulative_count.mp4", create_cumulative_count_video),
        ("state_tracking_occlusion.mp4", create_state_tracking_occlusion_video),
        ("backward_inference_occlusion.mp4", create_backward_inference_video),
    ]

    sft_data, rl_data = [], []
    for idx, (filename, func) in enumerate(videos_to_generate, 1):
        video_path = os.path.join(BASE_OUTPUT_DIR, filename)
        print(f"\n{'='*60}\n📹 Generating video {idx}/{len(videos_to_generate)}: {filename}\n{'='*60}")
        try:
            metadata = func(video_path)
            duration = get_video_duration(video_path)
            options_text, correct_letter = create_multiple_choice_data(metadata["question"], metadata["answer"], metadata["distractors"])
            full_question_text = f"{metadata['question']}\n{options_text}"

            sft_entry = {"type": "intermedia_sft","messages": [{"role": "system","content": [{"type": "text", "text": "You are a helpful assistant.", "loss": False}]},{"role": "user","content": [{"type": "video","uuid": str(uuid.uuid4()),"video": video_path,"width": metadata["width"],"height": metadata["height"],"duration": duration,"fps": metadata["fps"],"timestamp_fmt": "seconds","has_bisec_tar": False},{"type": "text", "text": full_question_text, "loss": False}]},{"role": "assistant","content": [{"type": "text", "text": correct_letter, "loss": True}]}],"extra": {"extra": {"id": idx, "doc_id": idx,"task": metadata.get("task_type", "Unknown Task"),"data_type": "video","data_source": "Synthetic_Video","original_video_path": video_path}}}
            sft_data.append(sft_entry)

            rl_entry = {"problem_id": idx,"problem": f"<video>{full_question_text}","data_type": "video","problem_type": metadata.get("task_type", "Unknown Task"),"videos": [video_path],"data_source": "Synthetic_Video","answer": correct_letter}
            rl_data.append(rl_entry)
        except Exception as e:
            print(f"❌ Error generating {filename} or its metadata: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}\n✨ All videos generated successfully!\n{'='*60}\n")
    sft_output_path = os.path.join(BASE_OUTPUT_DIR, "sft_data.jsonl")
    rl_output_path = os.path.join(BASE_OUTPUT_DIR, "rl_data.jsonl")
    write_jsonl(sft_data, sft_output_path)
    write_jsonl(rl_data, rl_output_path)

if __name__ == "__main__":
    generate_all_videos_and_metadata()
