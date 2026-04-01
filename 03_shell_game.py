"""Video type: Shell Game shell game/object tracking video
Description: A video of an object tracking game with container swapping on a 4x4 grid, containing the following sub-genres:
  Prediction mode (forward tracking):
  - basic_shell_game: single object tracking
  - multi_object_parallel_tracking: Multiple objects parallel tracking
  - unique_object_tracking: unique object tracking
  - return_to_origin: Return to original position to judge
  - state_change_count: position change count
  - swap_recognition: swap recognition
  - waypoint_query: path query
  - operation_count: operation count statistics
  - property_transformation: property transformation tracking
  - object_property_binding: object property binding
  Backtracking mode (reverse reasoning):
  - retro_track_to_origin: Reverse tracking starting position
  - retro_identify_from_origin: Reversely identify the source of an object"""

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

# --- 1. Configuration (global configuration) ---

# [--- MODIFIED ---] Please make sure this path is correct
plt.rcParams['animation.ffmpeg_path'] = "/home/data/songtao/MedEvalKit/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

# [--- NEW ---] Generate by type
NUM_PER_TYPE = 60
NUM_WORKERS = 64 # Keep 64 as per your requirement

BASE_OUTPUT_DIR = "/cpfs02/user/songtao.jst/reasoning_videos/RL/ball_swicth_V2/" # [--- MODIFIED ---]
OUTPUT_VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SFT_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "sft.jsonl")
RL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "rl.jsonl")
METADATA_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata.jsonl")
VIDEO_PREFIX_IN_JSON = os.path.join(BASE_OUTPUT_DIR, "videos/")

GRID_SIZE = 4
FPS = 30
INITIAL_REVEAL_FRAMES = 60
MOVE_DURATION_FRAMES = 45
STATIC_DURATION_FRAMES = 15
FINAL_REVEAL_FRAMES = 90

# [--- MODIFIED ---] Removed fifth category (logic and rules)
# Mode A: Predictive mode (forward tracking)
PRED_CHALLENGE_TYPES = [
    'basic_shell_game',             # 1.1
    'multi_object_parallel_tracking', # 1.2
    'unique_object_tracking',       # 1.3
    'return_to_origin',             # 2.1
    'state_change_count',           # 2.2
    'swap_recognition',             # 2.3
    'waypoint_query',               # 3.1
    'operation_count',              # 3.2
    'property_transformation',    # 4.1 upgraded version
    'object_property_binding',      # 4.2
]

# Mode B: Inference mode (reverse reasoning)
RETRO_CHALLENGE_TYPES = [
    'retro_track_to_origin',      # 6.1 "Where is the beginning of the object that ends in c3?"
    'retro_identify_from_origin'    # 6.2 "The object that starts at a1, which one ends at it?"
]

# [--- NEW ---] Merge all types
ALL_CHALLENGE_TYPES = PRED_CHALLENGE_TYPES + RETRO_CHALLENGE_TYPES

COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange']
SHAPES = ['circle', 'square', 'triangle']


# --- 2. Helper Functions ---
# [--- Your previous fix is ​​included ---]
def get_grid_map():
    """Generate a mapping of grid coordinates and labels"""
    coord_map = {}    # 'a1' -> (0.5, 3.5)
    pos_to_idx = {}   # 'a1' -> (0, 0)

    row_labels = [chr(ord('a') + i) for i in range(GRID_SIZE)]
    col_labels = [str(i + 1) for i in range(GRID_SIZE)]

    for r_idx, r_label in enumerate(row_labels):
        for c_idx, c_label in enumerate(col_labels):
            key = f"{r_label}{c_label}"
            coord = (c_idx + 0.5, GRID_SIZE - r_idx - 0.5)
            idx = (c_idx, r_idx)
            coord_map[key] = coord
            pos_to_idx[key] = idx

    coord_map_rev = {v: k for k, v in coord_map.items()}
    return coord_map, coord_map_rev, pos_to_idx

COORD_MAP, COORD_MAP_REV, POS_TO_IDX = get_grid_map()
ALL_POSITIONS = list(COORD_MAP.keys()) # [--- NEW ---] Global location

def get_pos_of_object(state, obj_id):
    """Find the location of the object with the specified ID in the state dictionary"""
    for pos, obj in state.items():
        if obj and obj['id'] == obj_id:
            return pos
    return None

def generate_position_options(correct_answer, all_positions, distract_from=None, none_option=None):
    """Build location options"""
    if not correct_answer:
        correct_answer = "None"

    options = [correct_answer]
    possible_decoys = set(all_positions) - {correct_answer}

    if distract_from:
        possible_decoys.update(distract_from)
        possible_decoys.discard(correct_answer)

    if none_option and correct_answer != none_option:
        options.append(none_option)

    num_decoys_needed = 4 - len(options)

    if possible_decoys:
        options.extend(random.sample(list(possible_decoys), min(len(possible_decoys), num_decoys_needed)))

    while len(options) < 4:
        available_decoys = list(set(all_positions) - set(options))
        if not available_decoys:
            options.append(f"Decoy_{len(options)}") # spare
        else:
            new_decoy = random.choice(available_decoys)
            options.append(new_decoy)

    random.shuffle(options)
    return options

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


# --- 3. [NEW] Biased Operation Generator ---

def generate_biased_operations(initial_state, num_ops, relevant_ids=None, relevant_pos=None):
    """Generate a sequence of operations and prioritize moving "relevant" objects or locations."""
    operations = []
    sim_state = initial_state.copy()

    relevant_ids = relevant_ids or set()
    relevant_pos = relevant_pos or set()

    PROB_MOVE_RELEVANT = 0.7 # 70% chance of forcing movement of related items

    for _ in range(num_ops):
        p1, p2 = None, None

        # Find the *current* location of the relevant object
        current_relevant_obj_pos = set()
        for obj_id in relevant_ids:
            pos = get_pos_of_object(sim_state, obj_id)
            if pos:
                current_relevant_obj_pos.add(pos)

        all_relevant_targets = current_relevant_obj_pos.union(relevant_pos)

        if all_relevant_targets and random.random() < PROB_MOVE_RELEVANT:
            # --- Biased movement ---
            p1 = random.choice(list(all_relevant_targets))
            possible_p2 = list(set(ALL_POSITIONS) - {p1})
            if possible_p2:
                p2 = random.choice(possible_p2)
            else:
                p1, p2 = random.sample(ALL_POSITIONS, 2) # spare
        else:
            # --- Random movement ---
            p1, p2 = random.sample(ALL_POSITIONS, 2)

        op = ('swap', p1, p2)
        operations.append(op)

        # Simulate this step
        obj1 = sim_state.pop(p1, None)
        obj2 = sim_state.pop(p2, None)
        if obj1:
            sim_state[p2] = obj1
        if obj2:
            sim_state[p1] = obj2

    return operations


# --- 4. Ground Truth Simulation Engine (core: truth simulator) ---
def simulate_operations(initial_state, operations, rules={}):
    """All operations are simulated, returning the final state and complete history.
    (Upgraded to track shape changes)"""
    current_state = {k: v.copy() for k, v in initial_state.items()}

    history = {
        'states': [current_state.copy()],
        'paths': {obj['id']: [pos] for pos, obj in initial_state.items() if obj},
        'colors': {obj['id']: [obj['color']] for pos, obj in initial_state.items() if obj},
        'shapes': {obj['id']: [obj['shape']] for pos, obj in initial_state.items() if obj}, # [--- NEW ---] Trace shapes
        'events': []
    }

    def update_history(obj_id, new_pos, new_color=None, new_shape=None): # [--- MODIFIED ---]
        if obj_id not in history['paths']:
            history['paths'][obj_id] = [new_pos]
            history['colors'][obj_id] = [new_color if new_color else 'unknown']
            history['shapes'][obj_id] = [new_shape if new_shape else 'unknown'] # [--- NEW ---]
        else:
            history['paths'][obj_id].append(new_pos)

            current_color = history['colors'][obj_id][-1]
            history['colors'][obj_id].append(new_color if new_color else current_color)

            current_shape = history['shapes'][obj_id][-1]
            history['shapes'][obj_id].append(new_shape if new_shape else current_shape) # [--- NEW ---]

    for op in operations:
        op_type, p1, p2, *args = op

        obj1 = current_state.pop(p1, None)
        obj2 = current_state.pop(p2, None)

        if obj1:
            current_state[p2] = obj1
        if obj2:
            current_state[p1] = obj2

        # --- Processing rules and special operations ---
        if op_type == 'magic_swap' and rules.get('rule_type') == 'property_transformation':
            target_obj = current_state.get(p2) # obj1 is now at p2
            if target_obj and target_obj['id'] == rules['target_obj_id']:
                new_color = rules['new_color']
                new_shape = rules['new_shape']
                target_obj['color'] = new_color
                target_obj['shape'] = new_shape
                history['events'].append(f"transform: {target_obj['id']} to {new_color} {new_shape}")

        # ---Update history (regular exchange) ---
        if obj1: update_history(obj1['id'], p2, obj1.get('color'), obj1.get('shape')) # [--- MODIFIED ---]
        if obj2: update_history(obj2['id'], p1, obj2.get('color'), obj2.get('shape')) # [--- MODIFIED ---]

        for pos, obj in current_state.items():
            if obj:
                obj_id = obj['id']
                if obj1 and obj_id == obj1['id']: continue
                if obj2 and obj_id == obj2['id']: continue

                update_history(obj_id, pos, obj.get('color'), obj.get('shape')) # [--- MODIFIED ---]

        history['states'].append(current_state.copy())

    final_state = {k: v for k, v in current_state.items() if v}
    return final_state, history


# --- 5. Challenge Generation Logic (Core: Challenge Generator) ---

# [--- MODIFIED ---] receives challenge_type as parameter
def generate_challenge_data(challenge_id, challenge_type):

    positions = ALL_POSITIONS

    initial_state = {}
    operations = []
    question = ""
    correct_answer = ""
    options = []

    # [--- Core logic modification ---]
    # The mode is determined by the challenge_type passed in
    mode = 'retrodictive' if challenge_type in RETRO_CHALLENGE_TYPES else 'predictive'

    metadata = {
        "challenge_type": challenge_type,
        "mode": mode  # 'predictive' 或 'retrodictive'
    }

    num_ops = random.randint(10, 15) # increased difficulty
    num_balls = random.randint(3, 6) # increased difficulty
    start_pos_list = random.sample(positions, num_balls)

    # --- Generic object creation ---
    all_objects = []

    available_colors = random.sample(COLORS, min(len(COLORS), num_balls + 2))
    available_shapes = random.sample(SHAPES, min(len(SHAPES), num_balls + 2))

    for i, pos in enumerate(start_pos_list):
        obj_id = f"obj_{i}"
        obj = {
            'id': obj_id,
            'color': available_colors[i % len(available_colors)],
            'shape': available_shapes[i % len(available_shapes)]
        }
        # Make sure the initial combination is unique
        while any(o['color'] == obj['color'] and o['shape'] == obj['shape'] for o in all_objects):
            obj['color'] = random.choice(COLORS)
            obj['shape'] = random.choice(SHAPES)

        initial_state[pos] = obj
        all_objects.append(obj)

    # --- [NEW] Define related items ---
    relevant_ids = set()
    relevant_pos = set()

    # --- 1.1 Classic "Three Immortals Return to the Cave" ---
    if challenge_type == 'basic_shell_game':
        num_balls = 1
        pos = random.sample(positions, 1)[0]
        obj = {'id': 'obj_0', 'color': 'red', 'shape': 'circle'}
        initial_state = {pos: obj}

        relevant_ids = {'obj_0'}
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)

        final_state, history = simulate_operations(initial_state, operations)
        correct_answer = get_pos_of_object(final_state, 'obj_0')
        question = "At the end of the video, where is the object located?"
        options = generate_position_options(correct_answer, positions, none_option="Nowhere")

    # --- 1.2 Multi-ball parallel tracking ---
    elif challenge_type == 'multi_object_parallel_tracking':
        color = 'blue'
        shape = 'circle'
        initial_state = {}
        all_objects = [] # reset
        for i, pos in enumerate(start_pos_list):
            obj_id = f'obj_{i}'
            initial_state[pos] = {'id': obj_id, 'color': color, 'shape': shape}
            all_objects.append(initial_state[pos])

        relevant_ids = {obj['id'] for obj in all_objects} # Track all balls
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        question = "Which of the following locations contains an object at the end?"
        correct_answer = random.choice(list(final_state.keys())) if final_state else "None"
        options = generate_position_options(correct_answer, positions, final_state.keys(), none_option="None")

    # --- 1.3 Unique Target Tracking ---
    elif challenge_type == 'unique_object_tracking':
        if not start_pos_list: return None
        target_obj = initial_state[start_pos_list[0]]
        target_obj['color'] = 'red'
        target_obj['shape'] = 'square'
        metadata['target_obj_id'] = target_obj['id']

        for i, pos in enumerate(start_pos_list[1:]):
            initial_state[pos]['color'] = 'blue'
            initial_state[pos]['shape'] = 'circle'

        relevant_ids = {target_obj['id']} # Only track red squares
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        correct_answer = get_pos_of_object(final_state, target_obj['id'])
        question = f"At the end of the video, where is the {target_obj['color']} {target_obj['shape']} located?"
        options = generate_position_options(correct_answer, positions, none_option="Nowhere")

    # --- 2.1 Homing query ---
    elif challenge_type == 'return_to_origin':
        if not initial_state: return None
        relevant_ids = {obj['id'] for obj in all_objects} # Track all balls
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        returned_balls_start_pos = []
        for start_pos, start_obj in initial_state.items():
            final_pos = get_pos_of_object(final_state, start_obj['id'])
            if start_pos == final_pos:
                returned_balls_start_pos.append(start_pos)

        # [--- FIX ---] Fixed question and answer wording
        question = "An object from which *starting position* returned to its original position?"
        if returned_balls_start_pos:
            correct_answer = random.choice(returned_balls_start_pos)
        else:
            correct_answer = "No object returned to its starting position"

        options = generate_position_options(correct_answer, positions, start_pos_list, none_option="No object returned to its starting position")

    # --- 2.2 Status change count ---
    elif challenge_type == 'state_change_count':
        if not initial_state: return None
        relevant_ids = {obj['id'] for obj in all_objects} # Track all balls
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        changed_count = 0
        for start_pos, start_obj in initial_state.items():
            final_pos = get_pos_of_object(final_state, start_obj['id'])
            if start_pos != final_pos:
                changed_count += 1

        question = "Compared to the initial state, how many objects are in a different position at the end?"
        correct_answer = str(changed_count)
        options = generate_numeric_options(correct_answer, max_val=num_balls)

    # --- 2.3 Position exchange recognition ---
    elif challenge_type == 'swap_recognition':
        if num_balls < 2:
            start_pos_list = random.sample(positions, 2)
            initial_state = {
                start_pos_list[0]: {'id': 'obj_0', 'color': 'red', 'shape': 'circle'},
                start_pos_list[1]: {'id': 'obj_1', 'color': 'blue', 'shape': 'square'}
            }
            all_objects = list(initial_state.values()) # reset

        p1_start = start_pos_list[0]
        p2_start = start_pos_list[1]
        obj1_id = initial_state[p1_start]['id']
        obj2_id = initial_state[p2_start]['id']

        relevant_ids = {obj1_id, obj2_id} # Just track these two balls
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        p1_final = get_pos_of_object(final_state, obj1_id)
        p2_final = get_pos_of_object(final_state, obj2_id)

        question = f"After the video, did the objects starting at {p1_start} and {p2_start} swap their positions?"
        if p1_final == p2_start and p2_final == p1_start:
            correct_answer = "Yes"
        else:
            correct_answer = "No"
        options = ["Yes", "No"]

    # --- 3.1 Query of route passing points ---
    elif challenge_type == 'waypoint_query':
        if not start_pos_list: return None
        target_obj = initial_state[start_pos_list[0]]
        target_obj_id = target_obj['id']
        target_start_pos = start_pos_list[0]
        metadata['target_obj_id'] = target_obj['id']

        relevant_ids = {target_obj_id} # tracking target ball
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        path = history['paths'][target_obj_id]
        if random.random() > 0.3 and len(path) > 2:
            waypoint = random.choice(path[1:-1])
            correct_answer = "Yes"
        else:
            waypoint = random.choice(list(set(positions) - set(path)))
            correct_answer = "No"

        question = f"Did the object starting from {target_start_pos} ever reach position {waypoint} during the process?"
        options = ["Yes", "No"]
        metadata['waypoint'] = waypoint

    # --- 3.2 Operation Count ---
    elif challenge_type == 'operation_count':
        target_pos = random.choice(positions)
        metadata['target_pos'] = target_pos

        relevant_pos = {target_pos} # track this location
        operations = generate_biased_operations(initial_state, num_ops, relevant_pos=relevant_pos)

        final_state, history = simulate_operations(initial_state, operations)

        count = 0
        for op in operations:
            if target_pos in op[1:]:
                count += 1

        question = f"In the video, how many times was the container at position {target_pos} moved in total?"
        correct_answer = str(count)
        options = generate_numeric_options(correct_answer, max_val=num_ops)

    # --- 4.1 Attribute transformation ---
    elif challenge_type == 'property_transformation':
        if not start_pos_list: return None
        target_obj = initial_state[start_pos_list[0]]
        target_obj_id = target_obj['id']
        target_start_pos = start_pos_list[0]

        metadata['rule_type'] = 'property_transformation'
        metadata['target_obj_id'] = target_obj_id

        distractor_objects = [obj for obj in all_objects if obj['id'] != target_obj_id]

        new_color, new_shape = None, None
        # 50% chance of becoming an existing object on the field
        if random.random() < 0.5 and distractor_objects:
            decoy_obj = random.choice(distractor_objects)
            new_color = decoy_obj['color']
            new_shape = decoy_obj['shape']
        else:
            # 50% chance of transforming into a completely new object
            all_init_combos = {(o['color'], o['shape']) for o in all_objects}
            new_color, new_shape = random.choice(COLORS), random.choice(SHAPES)
            while (new_color, new_shape) in all_init_combos:
                new_color, new_shape = random.choice(COLORS), random.choice(SHAPES)

        metadata['new_color'] = new_color
        metadata['new_shape'] = new_shape

        relevant_ids = {target_obj_id} # tracking target ball
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)

        # [--- NEW ---] Smart replacement with magic_swap
        magic_swap_idx = -1
        current_pos = target_start_pos
        sim_path = [current_pos]
        for i, op in enumerate(operations):
            op_type, p1, p2 = op[0], op[1], op[2]
            if p1 == current_pos:
                current_pos = p2
                sim_path.append(current_pos)
                magic_swap_idx = i
                break
            elif p2 == current_pos:
                current_pos = p1
                sim_path.append(current_pos)
                magic_swap_idx = i
                break

        if magic_swap_idx != -1:
             op = operations[magic_swap_idx]
             operations[magic_swap_idx] = ('magic_swap', op[1], op[2])
        else:
             # Backup: if (impossibly) not moved once, replace randomly
             op_idx = random.randint(0, num_ops-1)
             operations[op_idx] = ('magic_swap', operations[op_idx][1], operations[op_idx][2])

        final_state, history = simulate_operations(initial_state, operations, rules=metadata)

        final_color = history['colors'][target_obj_id][-1]
        final_shape = history['shapes'][target_obj_id][-1]

        question = f"Note: In this video, an object's properties (color or shape) may change. What are the final properties of the object that started at {target_start_pos}?"
        correct_answer = f"{final_color} {final_shape}"

        options = {correct_answer}
        options.add(f"{target_obj['color']} {target_obj['shape']}") # initial properties
        options.update(f"{o['color']} {o['shape']}" for o in distractor_objects) # Other object properties
        while len(options) < 4:
            options.add(f"{random.choice(COLORS)} {random.choice(SHAPES)}")

        options = list(options)
        if correct_answer not in options:
            options[0] = correct_answer
        random.shuffle(options)
        options = options[:4]

    # --- 4.2 Object-Property Binding ---
    elif challenge_type == 'object_property_binding':
        if not all_objects: return None
        target_obj = random.choice(all_objects)
        target_obj_id = target_obj['id']
        metadata['target_obj_id'] = target_obj['id']

        relevant_ids = {target_obj_id} # tracking target ball
        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        correct_answer = get_pos_of_object(final_state, target_obj['id'])
        question = f"At the end of the video, where is the {target_obj['color']} {target_obj['shape']} located?"
        options = generate_position_options(correct_answer, positions, none_option="Nowhere")

    # --- [NEW] 6.1 Inference mode: trace to starting point ---
    elif challenge_type == 'retro_track_to_origin':
        if not all_objects: return None

        target_obj = random.choice(all_objects)
        relevant_ids = {target_obj['id']} # tracking target ball

        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        final_pos = get_pos_of_object(final_state, target_obj['id'])
        initial_pos = get_pos_of_object(initial_state, target_obj['id'])

        if not final_pos or not initial_pos: return None

        question = f"At the end of the video, the {target_obj['color']} {target_obj['shape']} is at {final_pos}. Where was this object at the start?"
        correct_answer = initial_pos
        options = generate_position_options(correct_answer, positions)

    # --- [NEW] 6.2 Inference pattern: recognition from starting point ---
    elif challenge_type == 'retro_identify_from_origin':
        if not start_pos_list: return None

        target_start_pos = random.choice(start_pos_list) # Choose a starting point with an object
        target_obj = initial_state[target_start_pos]
        relevant_ids = {target_obj['id']} # tracking target ball

        operations = generate_biased_operations(initial_state, num_ops, relevant_ids)
        final_state, history = simulate_operations(initial_state, operations)

        question = f"Which object (as seen in the final reveal) started at position {target_start_pos}?"
        correct_answer = f"The {target_obj['color']} {target_obj['shape']}"

        # Construction options
        options = {correct_answer}
        decoys = set(f"The {o['color']} {o['shape']}" for o in all_objects if o['id'] != target_obj['id'])
        while len(options) < 4 and decoys:
            options.add(decoys.pop())
        i = 1
        while len(options) < 4:
            options.add(f"A {random.choice(COLORS)} {random.choice(SHAPES)}")
            i += 1
        options = list(options)
        if correct_answer not in options:
            options[0] = correct_answer
        random.shuffle(options)
        options = options[:4]

    # --- Final packaging ---

    none_strings = {"No object returned", "None", "Nowhere", "No object returned to its starting position"}
    if not correct_answer or (correct_answer in none_strings and correct_answer not in options):
         if "No object returned to its starting position" in options:
             correct_answer = "No object returned to its starting position"
         elif "No object returned" in options:
             correct_answer = "No object returned"
         elif "None" in options:
             correct_answer = "None"
         elif "Nowhere" in options:
             correct_answer = "Nowhere"
         elif "No" in options:
             correct_answer = "No"
         elif len(options) > 0:
             correct_answer = options[0]
         else:
             correct_answer = "Error"

    if correct_answer not in options:
        if challenge_type == 'waypoint_query' or challenge_type == 'swap_recognition':
             options = [correct_answer, "Yes" if correct_answer == "No" else "No"]
        elif correct_answer in none_strings and any(s in options for s in none_strings):
             pass
        elif len(options) >= 4:
             options[random.randint(0, 3)] = correct_answer
             random.shuffle(options)
        else:
             options.append(correct_answer)
             while len(options) < 4: options.append(f"Decoy_{len(options)}")

    video_filename = f"{challenge_type}_{challenge_id}.mp4"
    return {
        "id": challenge_id, "challenge_type": challenge_type, "grid_size": GRID_SIZE,
        "initial_state": initial_state, "operations": operations, "final_state": final_state,
        "question": question, "options": options, "correct_answer_value": correct_answer,
        "correct_answer_label": chr(ord('A') + options.index(correct_answer)),
        "video_filename": video_filename,

        ## --- MODIFICATION (Timestamp) --- ##
        # (Make sure the 'videos' key exists and is a list)
        "video_filepath": os.path.join(OUTPUT_VIDEO_DIR, video_filename), # Reserve video_filepath for use by vg
        "videos": [os.path.join(OUTPUT_VIDEO_DIR, video_filename)],
        ## --- MODIFICATION END --- ##

        "metadata": metadata
    }


# --- 6. Video Generation Engine ---
class VideoGenerator:
    ## --- MODIFICATION (Timestamp) --- ##
    # (Added self.events_timeline_ms in __init__)
    ## -------------------------------- ##
    def __init__(self, challenge_data):
        self.data = challenge_data
        self.metadata = challenge_data.get('metadata', {})
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.dynamic_patches = []

        # --- New: Precompute "millisecond (ms) status log" (only keep key actions) ---
        self.events_timeline_ms = []
        current_time_ms = 0.0

        # Convert from frames to milliseconds
        reveal_duration_ms = (INITIAL_REVEAL_FRAMES / FPS) * 1000
        move_duration_ms = (MOVE_DURATION_FRAMES / FPS) * 1000
        static_duration_ms = (STATIC_DURATION_FRAMES / FPS) * 1000
        final_reveal_duration_ms = (FINAL_REVEAL_FRAMES / FPS) * 1000

        mode = self.metadata.get('mode', 'predictive')

        # 1. Record the initial impression event (if any)
        if mode == 'predictive':
            start_ms = int(current_time_ms)
            current_time_ms += reveal_duration_ms
            end_ms = int(current_time_ms)
            self.events_timeline_ms.append({
                "event_type": "initial_reveal",
                "initial_state": self.data['initial_state'],
                "start_ms": start_ms,
                "end_ms": end_ms
            })
        else:
            # Otherwise, the time is still advanced, but the event is not logged
            current_time_ms += reveal_duration_ms

        # 2. Log all 'swap' operations
        for op in self.data['operations']:
            op_type, p1, p2, *args = op

            start_ms = int(current_time_ms)
            current_time_ms += move_duration_ms
            end_ms = int(current_time_ms)

            self.events_timeline_ms.append({
                "event_type": op_type, # 'swap' or 'magic_swap'
                "pos1": p1,
                "pos2": p2,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

            # Advance pause time (but don't log it)
            current_time_ms += static_duration_ms

        # 3. Record the final display event (if any)
        if mode == 'retrodictive':
            start_ms = int(current_time_ms)
            current_time_ms += final_reveal_duration_ms
            end_ms = int(current_time_ms)
            self.events_timeline_ms.append({
                "event_type": "final_reveal",
                "final_state": self.data['final_state'],
                "start_ms": start_ms,
                "end_ms": end_ms
            })
        else:
            # Otherwise, the time is still advanced, but the event is not logged
            current_time_ms += final_reveal_duration_ms
        # --- End of new addition ---
    ## --- MODIFICATION END --- ##


    def _draw_background_and_static_containers(self, exclude_containers=None, container_alpha=1.0):
        # (This function remains unchanged)
        exclude_containers = exclude_containers or set()
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, GRID_SIZE); self.ax.set_ylim(0, GRID_SIZE)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                color = '#FFFBEA' if (r + c) % 2 == 0 else '#E8E4D5' # beige/cream
                self.ax.add_patch(patches.Rectangle((c, r), 1, 1, color=color, zorder=0))

        row_labels, col_labels = [chr(ord('a')+i) for i in range(GRID_SIZE)], [str(i+1) for i in range(GRID_SIZE)]
        self.ax.set_xticks(np.arange(GRID_SIZE) + 0.5); self.ax.set_xticklabels(col_labels)
        self.ax.set_yticks(np.arange(GRID_SIZE) + 0.5); self.ax.set_yticklabels(reversed(row_labels))
        self.ax.tick_params(left=False, bottom=False)

        for key, pos in COORD_MAP.items():
            if isinstance(key, str) and key not in exclude_containers:
                self._draw_container_at_pos(pos, alpha=container_alpha)

    def _draw_container_at_pos(self, center_pos, alpha=1.0):
        # (This function remains unchanged)
        cx, cy = center_pos
        size = 0.7
        container = patches.Rectangle(
            (cx - size/2, cy - size/2), size, size,
            facecolor='#606060', edgecolor='black', lw=1.5, zorder=5, alpha=alpha
        )
        self.dynamic_patches.append(self.ax.add_patch(container))

    def _draw_object(self, pos_key, obj_dict):
        # (This function remains unchanged)
        cx, cy = COORD_MAP[pos_key]
        color = obj_dict['color']
        shape = obj_dict['shape']
        patch = None

        if shape == 'circle':
            patch = patches.Circle((cx, cy), radius=0.18, facecolor=color, zorder=2)
        elif shape == 'square':
            patch = patches.Rectangle((cx-0.18, cy-0.18), 0.36, 0.36, facecolor=color, zorder=2)
        elif shape == 'triangle':
            patch = patches.Polygon([(cx, cy+0.2), (cx-0.2, cy-0.15), (cx+0.2, cy-0.15)], facecolor=color, zorder=2)

        if patch:
            self.dynamic_patches.append(self.ax.add_patch(patch))

    def _update(self, frame):
        # (This function remains unchanged)
        [p.remove() for p in self.dynamic_patches]; self.dynamic_patches = []

        total_ops_frames = len(self.data['operations']) * (MOVE_DURATION_FRAMES + STATIC_DURATION_FRAMES)
        ops_end_frame = INITIAL_REVEAL_FRAMES + total_ops_frames

        # [--- New logic ---] Check video mode
        mode = self.data['metadata'].get('mode', 'predictive')

        # Phase 1: Initial rollout (conditional)
        if frame < INITIAL_REVEAL_FRAMES:
            if mode == 'predictive':
                # Prediction mode: shows initial state
                self._draw_background_and_static_containers(container_alpha=0.6) # transparent
                for pos, obj in self.data['initial_state'].items():
                    if obj: self._draw_object(pos, obj)
            else:
                # Inference mode: Hide initial state
                self._draw_background_and_static_containers(container_alpha=1.0) # opaque
            return

        # Stage 3: Final presentation (conditional)
        if frame >= ops_end_frame:
            if mode == 'retrodictive':
                # Inference mode: display the final state
                self._draw_background_and_static_containers(container_alpha=0.6) # transparent
                for pos, obj in self.data['final_state'].items():
                    if obj: self._draw_object(pos, obj)
            else:
                # Prediction mode: Hide final state
                self._draw_background_and_static_containers(container_alpha=1.0) # opaque
            return

        # Phase 2: Exchange process (always opaque)
        adj_frame = frame - INITIAL_REVEAL_FRAMES
        op_idx = adj_frame // (MOVE_DURATION_FRAMES + STATIC_DURATION_FRAMES)

        if op_idx >= len(self.data['operations']):
            self._draw_background_and_static_containers(container_alpha=1.0) # opaque
            return

        frame_in_op = adj_frame % (MOVE_DURATION_FRAMES + STATIC_DURATION_FRAMES)

        op = self.data['operations'][op_idx]
        op_type, p1, p2, *args = op

        # 2a: On the move
        if frame_in_op < MOVE_DURATION_FRAMES:
            progress = frame_in_op / (MOVE_DURATION_FRAMES - 1) if MOVE_DURATION_FRAMES > 1 else 1.0

            exclude_containers = {p1, p2}
            pairs_to_animate = [ (p1, p2, progress) ]

            self._draw_background_and_static_containers(exclude_containers=exclude_containers, container_alpha=1.0) # opaque

            for (pos1, pos2, prog) in pairs_to_animate:
                if pos1 not in COORD_MAP or pos2 not in COORD_MAP: continue
                start_pos, end_pos = np.array(COORD_MAP[pos1]), np.array(COORD_MAP[pos2])
                self._draw_container_at_pos(start_pos + (end_pos - start_pos) * prog, alpha=1.0)
                self._draw_container_at_pos(end_pos + (start_pos - end_pos) * prog, alpha=1.0)

        # 2b: moving gap (still frame)
        else:
            self._draw_background_and_static_containers(container_alpha=1.0) # opaque

    def generate_video(self):
        # (This function remains unchanged)
        total_frames = INITIAL_REVEAL_FRAMES + len(self.data['operations']) * (MOVE_DURATION_FRAMES + STATIC_DURATION_FRAMES) + FINAL_REVEAL_FRAMES

        ani = animation.FuncAnimation(self.fig, self._update, frames=total_frames, interval=1000/FPS)

        writer = animation.writers['ffmpeg'](fps=FPS, metadata=dict(artist='Gemini'), bitrate=2000)

        try:
            ani.save(self.data['video_filepath'], writer=writer, dpi=120)
        finally:
            plt.close(self.fig) # Regardless of success or failure, close the graph to prevent memory leaks

# --- 7. Parallel Processing Worker (multi-process worker) ---
# [--- MODIFIED ---] receive challenge_type
def process_single_challenge(challenge_id, challenge_type):
    try:
        challenge_data = generate_challenge_data(challenge_id, challenge_type)

        # [---Fixed ---] Catch all possible build failures
        if not challenge_data:
            return (challenge_id, f"Failed to generate challenge data (e.g., generation failed for type {challenge_type})")

        # [--- Check for "5 -> 4" BUG ---]
        if challenge_data['metadata']['mode'] == 'predictive' and\
           len(challenge_data['initial_state']) != len(challenge_data['final_state']):
            return (challenge_id, f"CRITICAL BUG: Object count mismatch. Start: {len(challenge_data['initial_state'])}, End: {len(challenge_data['final_state'])}")

        ## --- MODIFICATION (Timestamp) --- ##
        # (create vg here, it now contains events_timeline_ms internally)
        vg = VideoGenerator(challenge_data)
        vg.generate_video()

        # (Get self.events_timeline_ms status log)
        video_events_timeline_ms = vg.events_timeline_ms
        ## --- MODIFICATION END --- ##

        # --- Formatting SFT and RL ---
        question_text = challenge_data['question'] + "\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(challenge_data['options'])])

        # [--- MODIFIED ---] use challenge_data['videos'] (it is now a list)
        video_server_path = os.path.join(VIDEO_PREFIX_IN_JSON, challenge_data['video_filename']).replace("\\", "/")
        videos_list = [video_server_path]

        sft = {"type":"intermedia_sft", "messages":[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant.","loss":False}]},{"role":"user","content":[{"type":"video","uuid":str(uuid.uuid4()),"video":video_server_path},{"type":"text","text":question_text,"loss":False}]},{"role":"assistant","content":[{"type":"text","text":challenge_data['correct_answer_label'],"loss":True}]}],"extra":{"id":challenge_data['id'],"task":challenge_data['challenge_type'],"data_type":"video","data_source":"Synthetic_Video_Shell_Game","original_video_path":video_server_path}}

        ## --- MODIFICATION (Timestamp) --- ##
        # (Create a unified metadata dictionary with the same format as other scripts)
        ## -------------------------------- ##
        metadata = {
            # RL-style format
            "problem_id": challenge_data['id'],
            "problem": f"<video>\n{question_text}",
            "data_type": "video",
            "problem_type": challenge_data['challenge_type'],
            "videos": videos_list, # [--- MODIFIED ---] Make sure it is a list
            "data_source": "Synthetic_Video_Shell_Game",
            "answer": challenge_data['correct_answer_label'],
            "grid_size": challenge_data['grid_size'],

            # Detailed ground truth information
            "ground_truth_details": {
                "initial_state": challenge_data['initial_state'],
                "operations": challenge_data['operations'],
                "final_state": challenge_data['final_state'],
                "metadata": challenge_data['metadata'], # Contains mode, target_pos, etc.
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

        # (We now return sft, metadata, metadata)
        # (This way the contents of rl.jsonl and metadata.jsonl will be consistent)
        return sft, metadata, metadata
    except Exception as e:
        # Catch all other exceptions (e.g. memory exhausted, ffmpeg not found)
        return (challenge_id, f"Error in processing: {e} for type {challenge_type}\n{traceback.format_exc()}") # [--- ADDED ---]

# --- 8. Main Execution Logic (main execution logic) ---
def main():
    # [--- MODIFIED ---] Use NUM_PER_TYPE
    total_jobs = len(ALL_CHALLENGE_TYPES) * NUM_PER_TYPE
    print(f"Starting dataset generation...")
    print(f"Base output directory: {BASE_OUTPUT_DIR}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Challenge Types: {len(ALL_CHALLENGE_TYPES)}")
    print(f"Videos per Type: {NUM_PER_TYPE}")
    print(f"Total Videos: {total_jobs}")

    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        job_id = 0
        # [--- MODIFIED ---] Submit tasks in a loop
        for challenge_type in ALL_CHALLENGE_TYPES:
            for _ in range(NUM_PER_TYPE):
                # Pass job_id and challenge_type
                future = executor.submit(process_single_challenge, job_id, challenge_type)
                futures[future] = (job_id, challenge_type)
                job_id += 1

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_jobs, desc="Generating Challenges"):
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
        print("These tasks failed and did not hang:")
        for task_id, err in failed:
            print(f"  - Task {task_id}: {err}")

if __name__ == "__main__":
    main()
