"""Video type: Card Pile tracking video
Description: Generate synthetic video tracking poker deck operations
  - card_pile_tracking: Track the final state of the card pile after the add_top (top addition) and remove_bottom (bottom removal) operations
  3-5 cards are initially displayed. After 8-12 operations, the observer needs to track the order of the remaining cards."""

import os
import json
import random
import uuid
import shutil
import concurrent.futures
import traceback # [--- ADDED ---] Import traceback for more detailed error logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm


plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

# [--- NEW ---] Generate by type
NUM_VIDEOS_TO_GENERATE = 500
NUM_WORKERS = 64 # Adjust according to your CPU core number

BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/Card_Pile_V2/"
OUTPUT_VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SFT_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "sft.jsonl")
RL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "rl.jsonl")
METADATA_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata.jsonl")
VIDEO_PREFIX_IN_JSON = os.path.join(BASE_OUTPUT_DIR, "videos/")

FPS = 30
INITIAL_REVEAL_FRAMES = 150 # Initial display (5 seconds), add time for reading
OP_ANIMATION_FRAMES = 60    # Animation for each operation (2 seconds)
OP_STATIC_FRAMES = 30       # Pause between each operation (1 second)
FINAL_PAUSE_FRAMES = 60     # final pause

# --- 2. Card Deck Helpers (card auxiliary function) ---

SUITS = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}
SUIT_COLORS = {'H': 'red', 'D': 'red', 'C': 'black', 'S': 'black'}
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
FULL_DECK = [f"{r}{s}" for s in SUITS for r in RANKS]

def format_card(card_str):
    """ 'AC' -> 'Ace of Clubs' """
    rank_str, suit_str = card_str[:-1], card_str[-1]
    rank_map = {'A': 'Ace', 'K': 'King', 'Q': 'Queen', 'J': 'Jack'}
    rank_name = rank_map.get(rank_str, rank_str)
    suit_name = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'}[suit_str]
    return f"{rank_name} of {suit_name}"

def format_card_short(card_str):
    """ '10H' -> '10♥' """
    rank_str, suit_str = card_str[:-1], card_str[-1]
    return f"{rank_str}{SUITS[suit_str]}"

def format_pile_for_answer(pile):
    """ ['AC', '10H'] -> 'Ace of Clubs, 10 of Hearts' """
    if not pile:
        return "The pile is empty" # [--- NEW ---]
    return ", ".join(format_card(c) for c in pile)

# [--- MODIFIED ---] Upgraded distraction logic
def generate_answer_options(correct_pile, d1_initial, d2_reversed, d3_buggy_add):
    """Generate 4 smart distraction options"""
    correct_answer = format_pile_for_answer(correct_pile)
    options = {correct_answer}

    # Add smart distractions to ensure uniqueness
    options.add(format_pile_for_answer(d1_initial))
    if len(options) < 4:
        options.add(format_pile_for_answer(d2_reversed))
    if len(options) < 4:
        options.add(format_pile_for_answer(d3_buggy_add))

    # [--- NEW ---] If (extremely unlikely) the distractor and answer are duplicates,
    # Then use random cards to complete 4 options
    deck = random.sample(FULL_DECK, 10)
    while len(options) < 4:
        decoy_pile = [deck.pop(), deck.pop(), deck.pop()]
        options.add(format_pile_for_answer(decoy_pile))

    options = list(options)
    random.shuffle(options)
    return options, correct_answer


# --- 3. Simulation Engine (core: true value simulator) ---

def simulate_operations(initial_pile, operations):
    """Simulate all operations and return the final state.
    op_type: 'add_top', 'remove_bottom'"""
    pile = initial_pile.copy()

    for op_type, card in operations:
        if op_type == 'add_top':
            pile.insert(0, card) # Add at the beginning of the list
        elif op_type == 'remove_bottom':
            if pile: # Make sure the deck is not empty
                pile.pop() # remove end of list

    return pile

# --- 4. Challenge Generation Logic (Core: Challenge Generator) ---

# [--- MODIFIED ---] The entire function is rewritten to implement your new requirements
def generate_challenge_data(challenge_id):

    deck = random.sample(FULL_DECK, 52) # shuffle

    num_initial_cards = random.randint(3, 5)
    num_ops = random.randint(8, 12)

    initial_pile = [deck.pop() for _ in range(num_initial_cards)]

    operations = []
    sim_pile_size = num_initial_cards # for smart build operations

    for _ in range(num_ops):
        # 50/50 probability of adding or removing
        # But if the deck is empty, you must add
        if sim_pile_size == 0 or random.random() < 0.5:
            op_type = 'add_top'
            card = deck.pop()
            sim_pile_size += 1
        else:
            op_type = 'remove_bottom'
            # When removing, we don't need to know which card it was, but the simulator does
            card = None
            sim_pile_size -= 1

        operations.append((op_type, card))

    # --- Run a true simulation ---
    final_pile = simulate_operations(initial_pile, operations)

    # --- [NEW] Generating "wrong" distractors ---

    # Disruption 1: Reverse deck order (top/bottom confusion)
    buggy_pile_reversed = final_pile.copy()
    buggy_pile_reversed.reverse()

    # Interference 2: Operation error (add_top is mistaken for add_bottom)
    buggy_pile_add_bottom = initial_pile.copy()
    for op_type, card in operations:
        if op_type == 'add_top':
            buggy_pile_add_bottom.append(card) # <--- Wrong logic
        elif op_type == 'remove_bottom':
            if buggy_pile_add_bottom:
                buggy_pile_add_bottom.pop()

    # --- Generate questions and options ---
    options, correct_answer = generate_answer_options(
        final_pile,
        initial_pile, # Decoy 1: Initial state (no tracking)
        buggy_pile_reversed, # Decoy 2: In reverse order
        buggy_pile_add_bottom  # Decoy 3: Wrong operation
    )

    # [--- NEW ---] Optimized question text
    question = (
        "The video begins by showing an initial set of cards, "
        "which are then collected into a single pile from left to right "
        "(the leftmost card is on top, the rightmost is on the bottom). "
        "A series of operations (adding a card to the top, or removing a card from the bottom) "
        "is then performed. What are the cards remaining in the pile at the end, listed from top to bottom?"
    )

    video_filename = f"card_pile_challenge_{challenge_id}.mp4"
    data = {
        "id": challenge_id,
        "challenge_type": "card_pile_tracking",
        "question": question,
        "options": options,
        "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_filename": video_filename,
        "video_filepath": os.path.join(OUTPUT_VIDEO_DIR, video_filename),

        # The truth value passed to VideoGenerator
        "initial_pile": initial_pile,
        "operations": operations,
        "final_pile": final_pile
    }
    return data

# --- 5. Video Generation Engine ---
class VideoGenerator:
    ## --- MODIFICATION (Timestamp) --- ##
    # (Added self.events_timeline_ms in __init__)
    ## -------------------------------- ##
    def __init__(self, challenge_data):
        self.data = challenge_data
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.dynamic_patches = []

        # status tracking
        self.current_pile = self.data['initial_pile'].copy()

        # Animation constants
        self.pile_center = (4, 3)
        self.op_start_pos = (7, 1) # Where new cards appear
        self.op_end_pos = (1, 1)   # Remove the location where the card goes

        # --- New: Precompute "millisecond (ms) status log" (only keep key actions) ---
        self.events_timeline_ms = []
        current_time_ms = 0.0

        # Convert from frames to milliseconds
        reveal_duration_ms = (INITIAL_REVEAL_FRAMES / FPS) * 1000
        animation_duration_ms = (OP_ANIMATION_FRAMES / FPS) * 1000
        static_duration_ms = (OP_STATIC_FRAMES / FPS) * 1000

        # 1. Record the initial display event
        start_ms = int(current_time_ms)
        current_time_ms += reveal_duration_ms
        end_ms = int(current_time_ms)
        self.events_timeline_ms.append({
            "event_type": "initial_reveal",
            "initial_pile": self.data['initial_pile'], # key data
            "start_ms": start_ms,
            "end_ms": end_ms
        })

        # 2. Log all operations (requires an internal simulation to capture removed cards)
        sim_pile_for_logging = self.data['initial_pile'].copy()
        for op_type, card_from_op in self.data['operations']:
            start_ms = int(current_time_ms)
            current_time_ms += animation_duration_ms
            end_ms = int(current_time_ms)

            event_data = {
                "start_ms": start_ms,
                "end_ms": end_ms
            }

            if op_type == 'add_top':
                event_data["event_type"] = "add_top"
                event_data["card_id"] = card_from_op # Card is provided
                sim_pile_for_logging.insert(0, card_from_op) # Update simulation

            elif op_type == 'remove_bottom':
                event_data["event_type"] = "remove_bottom"
                if sim_pile_for_logging:
                    removed_card = sim_pile_for_logging.pop()
                    event_data["card_id"] = removed_card # Keep track of removed cards
                else:
                    event_data["card_id"] = None # Record a no-op

            self.events_timeline_ms.append(event_data)

            # Advance time (but do not log "pause" events)
            current_time_ms += static_duration_ms

        # 3. Final pause (not recorded)
        # --- End of new addition ---
    ## --- MODIFICATION END --- ##


    def _draw_background(self):
        """Drawn green poker table background"""
        self.ax.clear()
        self.ax.set_facecolor('#006442')
        self.ax.set_xlim(0, 8); self.ax.set_ylim(0, 6)
        self.ax.axis('off')

    def _draw_card(self, center_pos, card_str=None, face_down=False):
        """Draw a card at the specified location"""
        cx, cy = center_pos
        card_width, card_height = 1.0, 1.4

        if face_down:
            patch = patches.Rectangle(
                (cx - card_width/2, cy - card_height/2), card_width, card_height,
                facecolor='#FF69B4', edgecolor='black', lw=1, zorder=10
            )
        else:
            # card face up
            patch = patches.Rectangle(
                (cx - card_width/2, cy - card_height/2), card_width, card_height,
                facecolor='white', edgecolor='black', lw=1, zorder=10
            )
            # Draw text
            short_text = format_card_short(card_str)
            color = SUIT_COLORS[card_str[-1]]
            self.dynamic_patches.append(
                self.ax.text(cx, cy, short_text, ha='center', va='center', color=color, fontsize=18, weight='bold', zorder=11)
            )

        self.dynamic_patches.append(self.ax.add_patch(patch))
        return patch

    def _update(self, frame):
        # (This function remains unchanged)
        [p.remove() for p in self.dynamic_patches]; self.dynamic_patches = []
        self._draw_background()

        total_op_frames = len(self.data['operations']) * (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)
        op_start_frame = INITIAL_REVEAL_FRAMES
        op_end_frame = op_start_frame + total_op_frames

        # Phase 1: Initial presentation
        if frame < op_start_frame:
            self._draw_background()
            self.dynamic_patches.append(
                self.ax.text(self.pile_center[0], self.pile_center[1] + 1.5, "Initial Cards (Left is Top)", ha='center', va='center', color='white', fontsize=16)
            )

            # Expand and draw initial cards
            initial_pile = self.data['initial_pile']
            num_cards = len(initial_pile)
            for i, card_str in enumerate(initial_pile):
                cx = self.pile_center[0] - (num_cards/2 * 1.1) + (i * 1.1) + 0.5
                cy = self.pile_center[1]
                self._draw_card((cx, cy), card_str, face_down=False)

                # [--- NEW ---] Explicitly label top and bottom
                if i == 0:
                    self.dynamic_patches.append(
                        self.ax.text(cx, cy - 1.0, "Top", ha='center', va='center', color='white', fontsize=14)
                    )
                if i == num_cards - 1:
                    self.dynamic_patches.append(
                        self.ax.text(cx, cy - 1.0, "Bottom", ha='center', va='center', color='white', fontsize=14)
                    )
            return

        # Stage 3: Final pause
        if frame >= op_end_frame:
            self._draw_background()
            self._draw_card(self.pile_center, face_down=True)
            self.dynamic_patches.append(
                self.ax.text(self.pile_center[0], self.pile_center[1] + 1.5, f"Final Pile (Cards: {len(self.data['final_pile'])})", ha='center', va='center', color='white', fontsize=16)
            )
            return

        # Phase 2: Exchange Process
        adj_frame = frame - op_start_frame
        op_idx = adj_frame // (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)
        frame_in_op = adj_frame % (OP_ANIMATION_FRAMES + OP_STATIC_FRAMES)

        if op_idx >= len(self.data['operations']): return # Protect

        op_type, card_str = self.data['operations'][op_idx]

        # 2a: animation frame
        if frame_in_op < OP_ANIMATION_FRAMES:
            progress = frame_in_op / (OP_ANIMATION_FRAMES - 1)

            # Draw a static deck
            self._draw_card(self.pile_center, face_down=True)

            if op_type == 'add_top':
                # Moving card (up)
                start_pos = np.array(self.op_start_pos)
                end_pos = np.array(self.pile_center)
                anim_coord = start_pos + (end_pos - start_pos) * progress
                self._draw_card(anim_coord, card_str, face_down=False)

            elif op_type == 'remove_bottom':
                # [FIX] Simulate to previous step to get removed cards
                temp_pile = simulate_operations(self.data['initial_pile'], self.data['operations'][:op_idx])
                if temp_pile: # Make sure the previous step of the simulation is not empty
                    removed_card_str = temp_pile[-1] # it's the one that was removed
                    start_pos = np.array(self.pile_center)
                    end_pos = np.array(self.op_end_pos)
                    anim_coord = start_pos + (end_pos - start_pos) * progress
                    self._draw_card(anim_coord, removed_card_str, face_down=False)

        # 2b: Static frame (update status)
        else:
            # Simulation to this point
            self.current_pile = simulate_operations(self.data['initial_pile'], self.data['operations'][:op_idx+1])
            if self.current_pile: # [--- NEW ---] Only draw when there are cards in the deck
                self._draw_card(self.pile_center, face_down=True)

        # [--- MODIFIED ---] Removed "pile0"
        self.dynamic_patches.append(
            self.ax.text(self.pile_center[0], self.pile_center[1] + 1.5, f"Cards: {len(self.current_pile)}", ha='center', va='center', color='white', fontsize=16)
        )

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

        # --- Formatting SFT and RL ---
        question_text = challenge_data['question'] + "\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(challenge_data['options'])])
        video_server_path = os.path.join(VIDEO_PREFIX_IN_JSON, challenge_data['video_filename']).replace("\\", "/")

        sft = {"type":"intermedia_sft", "messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant.","loss":False}]},{"role":"user","content":[{"type":"video","uuid":str(uuid.uuid4()),"video":video_server_path},{"type":"text","text":question_text,"loss":False}]},{"role":"assistant","content":[{"type":"text","text":challenge_data['correct_answer_label'],"loss":True}]}],"extra":{"id":challenge_data['id'],"task":challenge_data['challenge_type'],"data_type":"video","data_source":"Synthetic_Video_Card_Pile","original_video_path":video_server_path}}

        # [--- MODIFIED START ---] (exactly the same as your previous format)
        # Create a metadata record containing both old and new formats
        metadata = {
            # RL-style format
            "problem_id": challenge_data['id'],
            "problem": f"<video>\n{question_text}",
            "data_type": "video",
            "problem_type": challenge_data['challenge_type'],
            "videos": [video_server_path],
            "data_source": "Synthetic_Video_Card_Pile",
            "answer": challenge_data['correct_answer_label'],

            # Detailed ground truth information
            "ground_truth_details": {
                "initial_pile": challenge_data['initial_pile'],
                "operations": challenge_data['operations'],
                "final_pile": challenge_data['final_pile'],
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
        # [--- MODIFIED END ---]

        return sft, metadata, metadata # [--- MODIFIED ---] Return uniformly
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
             # [--- MODIFIED ---] Make sure res[2] (metadata) exists
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
