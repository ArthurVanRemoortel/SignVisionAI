import functools
import os
from multiprocessing import Pool

import django
from pathlib import Path

from mediapipe.tasks.python.components.containers import Category
from mediapipe.tasks.python.vision import HandLandmarkerResult
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SignVisionSite.settings")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
django.setup()
from signvision_core.models import Hands, Word, Gesture, SignLanguage, GestureEntry, GestureEntryType

from typing import List
import cv2
from pathlib import Path
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame
import mediapipe as mp

MediapipeBaseOptions = mp.tasks.BaseOptions
MediapipeHandLandmarker = mp.tasks.vision.HandLandmarker
MediapipeHandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
MediapipeVisionRunningMode = mp.tasks.vision.RunningMode
MediapipeFaceLandmarker = mp.tasks.vision.FaceLandmarker
MediapipeFaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


def read_video_frames(video_path: Path) -> (float, [np.ndarray]):
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_array = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    frame_index = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        video_array[frame_index] = frame
        frame_index += 1
    cap.release()
    return fps, video_array


def create_gesture_frame(
        hand_landmarks: HandLandmarkerResult,
        face_landmarks: List[NormalizedLandmark] | None,
        left: bool,
        right: bool
) -> GestureFrame:
    left_landmarks = []
    right_landmarks = []
    found_hands = False

    handedness: [Category]
    for hand_landmarks, handedness in zip(
            hand_landmarks.hand_landmarks, hand_landmarks.handedness
    ):
        wrist_landmark = hand_landmarks[0]
        thumb_base_landmark = hand_landmarks[1]
        thumb_tip_landmark = hand_landmarks[4]
        index_base_landmark = hand_landmarks[5]
        index_tip_landmark = hand_landmarks[8]
        middle_base_landmark = hand_landmarks[9]
        middle_tip_landmark = hand_landmarks[12]
        ring_base_landmark = hand_landmarks[13]
        ring_tip_landmark = hand_landmarks[16]
        pink_base_landmark = hand_landmarks[17]
        pink_tip_landmark = hand_landmarks[20]

        landmark_groups = [
            wrist_landmark, thumb_base_landmark, thumb_tip_landmark, index_base_landmark, index_tip_landmark,
            middle_base_landmark, middle_tip_landmark, ring_base_landmark, ring_tip_landmark, pink_base_landmark,
            pink_tip_landmark
        ]
        coords = [Coordinate(landmark.x, landmark.y, landmark.z) for landmark in landmark_groups]
        # coords.append(Coordinate(mouth.x, mouth.y, mouth.z))
        if handedness[0].category_name == "Left" and left:
            left_landmarks = coords
            found_hands = True
        if handedness[0].category_name == "Right" and right:
            right_landmarks = coords
            found_hands = True

    if not left_landmarks:
        left_landmarks = [Coordinate.empty() for _ in range(len(HandFrame.__annotations__))]
    if not right_landmarks:
        right_landmarks = [Coordinate.empty() for _ in range(len(HandFrame.__annotations__))]

    mouth = face_landmarks[0] if found_hands and face_landmarks else Coordinate.empty()
    return GestureFrame(LEFT=HandFrame(*left_landmarks), RIGHT=HandFrame(*right_landmarks), MOUTH=Coordinate(mouth.x, mouth.y, mouth.z))


def detect_color_format(frame) -> str:
    num_channels = frame.shape[2] if len(frame.shape) == 3 else 1
    if num_channels == 3:
        if frame[0, 0, 0] == frame[0, 0, 2]:  # Check if blue and red channels are the same
            return "BGR"
        else:
            return "RGB"
    else:
        return "Grayscale"


def analyze_video(gesture: Gesture, video_path: Path) -> GestureDatasetEntry:
    fps, video_data = read_video_frames(video_path)
    hands_options = MediapipeHandLandmarkerOptions(
        base_options=MediapipeBaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=MediapipeVisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_options = MediapipeFaceLandmarkerOptions(
        base_options=MediapipeBaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=MediapipeVisionRunningMode.VIDEO,
        min_tracking_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    gesture_data = GestureDatasetEntry(name=video_path.name)

    with (MediapipeHandLandmarker.create_from_options(hands_options) as hands_landmarker,
          MediapipeFaceLandmarker.create_from_options(face_options) as face_landmarker):
        timestamp_ms = 0
        frame_interval = int(1000 / fps)
        for frame_n, frame_data in enumerate(video_data):
            # if frame_n == 0:
            #     print(video_path, detect_color_format(frame_data))
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_data)
            timestamp_ms += frame_interval
            hand_landmarker_result = hands_landmarker.detect_for_video(mp_frame, timestamp_ms)
            face_landmarker_result = face_landmarker.detect_for_video(mp_frame, timestamp_ms)
            if hand_landmarker_result.hand_landmarks:# and face_landmarker_result.face_landmarks:
                gesture_frame = create_gesture_frame(
                    hand_landmarker_result, face_landmarker_result.face_landmarks[0] if face_landmarker_result.face_landmarks else None,
                    left=True,
                    right=True)
                gesture_data.add_frame(gesture_frame)
            else:
                # Insert empty frame
                gesture_data.add_frame(GestureFrame.empty())
    return gesture_data


if __name__ == '__main__':
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    VGT = SignLanguage.objects.get(abbreviation='VGT')
    for gesture_folder in tqdm(list(Path('../../data/datasets/vgt-videos').iterdir())):
        gesture_name, handedness_string = gesture_folder.name.split('_')
        hand = Hands.BOTH
        if handedness_string[0] == '1' and handedness_string[1] == '0':
            hand = Hands.LEFT
        elif handedness_string[0] == '0' and handedness_string[1] == '1':
            hand = Hands.RIGHT
        word, _ = Word.objects.get_or_create(word=gesture_name, language=VGT)
        gesture, _ = Gesture.objects.get_or_create(word=word, hands=hand)
        # if not gesture.word.word.lower().__contains__('belg'):
        #     continue
        videos = []
        for video_path in list(gesture_folder.iterdir()):
            source_name = gesture_folder.name + "/" + video_path.name
            if GestureEntry.objects.filter(source_name=source_name).exists():
                continue
            videos.append(video_path)
            #break

        # for v in videos:
        #     analyze_video(gesture, v)

        with Pool(processes=12) as pool:
            gesture_data_results = pool.map(
                functools.partial(analyze_video, gesture),
                videos,
            )
            gesture_data_result: GestureDatasetEntry
            for i, gesture_data_result in enumerate(gesture_data_results):
                video_path = videos[i]
                source_name = gesture_folder.name + "/" + video_path.name
                if not gesture_data_result.frames:
                    print("Warning: No data found in " + source_name)
                    continue
                pop_rate = gesture_data_result.populated_rate()
                pop_count = len(gesture_data_result.populated_frames())
                if pop_rate < 0.2 and pop_count < 20:
                    print(f"Warning: {video_path} has a populated rate of {int(pop_rate*100)}%. It will not be saved.")
                    continue
                elif pop_rate < 0.2 and pop_count > 20:
                    print(f"Warning: {video_path} has a populated rate of {int(pop_rate*100)}% but still has {pop_count} frames. I might still be included.")
                    continue
                entry = GestureEntry.from_gesture_dataset_entry(gesture, gesture_data_result, source_name)
                entry.entry_type = GestureEntryType.MEDIAPIPE_VIDEO
                entry.save()


