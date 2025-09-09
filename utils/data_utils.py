import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import unicodedata

# --- Landmark Configuration ---

UPPER_BODY_POSE_INDICES = list(range(25))

MINIMAL_FACE_INDICES = sorted(list(set([
    0, 17, 61, 291, # Mouth anchors
    40, 39, 37, 270, 269, 267, # Upper lip contour
    1, # Nose tip
])))

NUM_POSE_LANDMARKS = len(UPPER_BODY_POSE_INDICES)
NUM_FACE_LANDMARKS = len(MINIMAL_FACE_INDICES)
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + (2 * NUM_HAND_LANDMARKS)
LANDMARK_DIM = 4

def extract_keypoints_structured(results):
    """Extracts keypoints into a structured [num_landmarks, 4] numpy array."""
    all_landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for i, idx in enumerate(UPPER_BODY_POSE_INDICES):
            all_landmarks[i] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z, landmarks[idx].visibility]

    if results.face_landmarks:
        landmarks = results.face_landmarks.landmark
        offset = NUM_POSE_LANDMARKS
        for i, idx in enumerate(MINIMAL_FACE_INDICES):
            all_landmarks[offset + i] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z, 1.0]

    if results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
        offset = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
        for i, res in enumerate(landmarks):
            all_landmarks[offset + i] = [res.x, res.y, res.z, 1.0]

    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        offset = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS
        for i, res in enumerate(landmarks):
            all_landmarks[offset + i] = [res.x, res.y, res.z, 1.0]

    return all_landmarks

def augment_keypoints_structured(sequence):
    """Applies augmentations to a structured sequence."""
    aug_sequence = sequence.copy()
    aug_sequence += np.random.normal(0, 0.005, size=aug_sequence.shape)
    scale_factor = np.random.uniform(0.9, 1.1)
    trans_x, trans_y = np.random.uniform(-0.05, 0.05, 2)
    aug_sequence[:, :, 0:2] *= scale_factor
    aug_sequence[:, :, 0] += trans_x
    aug_sequence[:, :, 1] += trans_y
    return aug_sequence.astype(np.float32)

def create_structured_npy_dataset(config):
    """Processes videos and saves standardized .npy files."""
    if os.path.exists(config['output_npy_folder']):
        print(f"Removing existing directory: {config['output_npy_folder']}")
        os.system(f"rm -rf {config['output_npy_folder']}")
    os.makedirs(config['output_npy_folder'], exist_ok=True)

    labels_file_path = config.get('labels_file_path')
    if not labels_file_path:
        for root, _, files in os.walk(config['base_data_folder']):
            if 'data.txt' in files:
                labels_file_path = os.path.join(root, 'data.txt')
                break
    
    if not labels_file_path:
        raise FileNotFoundError("data.txt not found in the base data folder.")

    video_to_label_map = {}
    with open(labels_file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                filename, label = line.split(',')
                video_to_label_map[filename.strip()] = label.strip()

    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    for video_filename, clean_label in tqdm(video_to_label_map.items(), desc="Processing Videos"):
        video_path = None
        for root, _, files in os.walk(config['base_data_folder']):
            if video_filename in files:
                video_path = os.path.join(root, video_filename)
                break
        
        if not video_path:
            print(f"Warning: Video '{video_filename}' not found. Skipping.")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_keypoints.append(extract_keypoints_structured(results))
        cap.release()

        if len(frame_keypoints) > 0:
            sequence = np.array(frame_keypoints)
            fixed_length_sequence = np.zeros((config['sequence_length'], TOTAL_LANDMARKS, LANDMARK_DIM))

            # --- FIX APPLIED HERE ---
            # Correctly compare the number of frames (sequence.shape[0]) with the target length.
            if sequence.shape[0] > config['sequence_length']:
                # Downsample the sequence if it's too long
                indices = np.linspace(0, sequence.shape[0] - 1, config['sequence_length'], dtype=int)
                fixed_length_sequence = sequence[indices]
            else:
                # Pad the sequence if it's too short
                fixed_length_sequence[:sequence.shape[0]] = sequence

            unique_id_parts = video_filename.split('.')[:-1]
            unique_id = ".".join(unique_id_parts)
            np.save(os.path.join(config['output_npy_folder'], f"{clean_label}_{unique_id}_original.npy"), fixed_length_sequence)

            for i in range(config['num_augmented_copies']):
                augmented_sequence = augment_keypoints_structured(fixed_length_sequence)
                np.save(os.path.join(config['output_npy_folder'], f"{clean_label}_{unique_id}_aug_{i+1}.npy"), augmented_sequence)

    holistic.close()
    print(f"\nCreated {len(os.listdir(config['output_npy_folder']))} standardized .npy files.")
    return config['output_npy_folder']

def get_label_from_npy_filename(filename):
    """Extracts the full multi-word label from .npy filenames."""
    base_name = os.path.basename(filename).replace('.npy', '')
    parts = base_name.split('_')
    label_parts = []
    # This logic assumes filename is like 'label_part1_label_part2_uniqueid_aug_1.npy'
    # It might need adjustment if your unique_id contains underscores.
    # A safer way might be to find the last parts like '_original' or '_aug_d'.
    # For now, this reflects the original logic.
    is_label_part = True
    temp_label = []
    for part in parts:
        # A simple heuristic to stop capturing label parts.
        # This might need to be more robust depending on your filenames.
        if part.startswith('vid') or part in ['original', 'aug']:
            is_label_part = False
        if is_label_part:
            temp_label.append(part)
        else:
            # Once we stop, we don't start again
            pass 
    
    # Rebuilding the label. This is a bit tricky and depends on conventions.
    # The original logic is kept here.
    for part in parts:
        if part.startswith('vid'): # Assuming 'vid' marks the start of the unique ID
            break
        label_parts.append(part)

    label_name = '_'.join(label_parts)
    normalized_label_name = unicodedata.normalize('NFC', label_name)
    return normalized_label_name