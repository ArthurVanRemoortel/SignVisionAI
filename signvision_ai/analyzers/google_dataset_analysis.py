import os
from pprint import pprint

import django
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm
import cv2
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SignVisionSite.settings")
django.setup()

import json
from pathlib import Path
from signvision_core.models import Word, SignLanguage, Country, Gesture, Hands
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, HandFrame, Coordinate


window_width = 800
window_height = 600
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 3

hand_landmarks = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20}
face_landmarks = {0}
coord_cols = ['x', 'y', 'z']
selected_landmarks = hand_landmarks.union(face_landmarks)


dataset_root = Path("D:/SL/asl-signs")
index_dict = json.load(open(dataset_root / "sign_to_prediction_index_map.json", "r"))
# load data from csv file
train_df: pandas.DataFrame = pandas.read_csv(dataset_root / "train.csv")
Asl = SignLanguage.objects.get(abbreviation="ASL")


def preview(data):
    do_exit = False
    while not do_exit:
        gesture_dataset_entry: GestureDatasetEntry = data
        gesture_dataset_entry.prepare()
        background = 255 * np.ones((window_height, window_width, 3), dtype=np.uint8)
        frame: GestureFrame
        for i, frame in enumerate(gesture_dataset_entry.frames):
            image = background.copy()
            for hand_name in ['LEFT', 'RIGHT']:
                hand_frame: HandFrame = frame.__getattribute__(hand_name)
                color = (0, 0, 255) if hand_name == 'RIGHT' else (255, 0, 0)
                if hand_frame.is_empty():
                    continue
                for landmark_name in hand_frame.__annotations__.keys():
                    landmark_value = hand_frame.__getattribute__(landmark_name)
                    # print(landmark_value.x, landmark_value.y)
                    center = (int(landmark_value.x * window_width), int(landmark_value.y * window_height))
                    cv2.circle(image, center, 10 if 'TIP' not in landmark_name else 4, color, -1)  # Draw a red circle at the coordinate

            if not frame.MOUTH.is_empty():
                cv2.circle(image, (int(frame.MOUTH.x * window_width), int(frame.MOUTH.y * window_height)), 20,
                           (255, 0, 0), -1)  # Draw a red circle at the coordinate

            cv2.imshow('Gesture Preview', image)

            if cv2.waitKey(5) & 0xFF == 27:
                do_exit = True
                break
    cv2.destroyAllWindows()


def process_dataset_word(row):
    parquet_file = row[0]
    word_data_df: pandas.DataFrame = pd.read_parquet(dataset_root / parquet_file, filters=[("landmark_index", "in", selected_landmarks)])
    sequence_id = row[2]
    word_name = row[3]
    entry_name = str(sequence_id) + "/" + word_name
    left_landmarks = {ll:[] for ll in hand_landmarks}
    right_landmarks = {rl:[] for rl in hand_landmarks}
    mouth_positions = []
    for frame_n, frame_data in word_data_df.iterrows():  # Some frames have a frozen hand.
        _, _, frame_type, landmark, x, y, z = frame_data
        print(frame_data) # Check frame numbers are correct.
        x = None if pd.isna(x) else x
        y = None if pd.isna(y) else y
        z = None if pd.isna(z) else z
        if landmark == 0 and frame_type == "face":
            mouth_positions.append(Coordinate(x, y, z))
        elif landmark in hand_landmarks and frame_type == "right_hand":
            right_landmarks[landmark].append(Coordinate(x, y, z))
        elif landmark in hand_landmarks and frame_type == "left_hand":
            left_landmarks[landmark].append(Coordinate(x, y, z))
    if len(mouth_positions) < 35:
        return
    gesture_data = GestureDatasetEntry()
    for frame_n, mouth in enumerate(mouth_positions):
        left_hand = HandFrame(*[v[frame_n] for l, v in  left_landmarks.items()])
        right_hand = HandFrame(*[v[frame_n] for l, v in  right_landmarks.items()])
        if not left_hand.is_empty() or not right_hand.is_empty():
            gesture_data.add_frame(GestureFrame(left_hand, right_hand, mouth))
    # print(word_name, mouth_positions.__len__(), gesture_data.frames.__len__())
    if gesture_data.frames.__len__() < 35:
        return
    return gesture_data
