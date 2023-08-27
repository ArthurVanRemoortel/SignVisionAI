import json
from pathlib import Path

import cv2
import requests
from mediapipe.tasks.python.components.containers import Category
from mediapipe.tasks.python.vision import HandLandmarkerResult
# from tqdm import tqdm
# from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
# from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from signvision_ai.analyzers.video_analysis import read_video_frames, create_gesture_frame
from signvision_ai.dataset import process_hand_openness, GestureDatasetEntry, process_hand_orientation

MediapipeBaseOptions = mp.tasks.BaseOptions
MediapipeHandLandmarker = mp.tasks.vision.HandLandmarker
MediapipeHandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
MediapipeVisionRunningMode = mp.tasks.vision.RunningMode
MediapipeFaceLandmarker = mp.tasks.vision.FaceLandmarker
MediapipeFaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


if __name__ == '__main__':
    hands_options = MediapipeHandLandmarkerOptions(
        base_options=MediapipeBaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=MediapipeVisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=.5,
        min_tracking_confidence=0.5,
    )
    face_options = MediapipeFaceLandmarkerOptions(
        base_options=MediapipeBaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=MediapipeVisionRunningMode.VIDEO,
        min_tracking_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    with (MediapipeHandLandmarker.create_from_options(hands_options) as hands_landmarker,
          MediapipeFaceLandmarker.create_from_options(face_options) as face_landmarker):

        cap = cv2.VideoCapture(0)
        timestamp_ms = 0

        gesture_data = GestureDatasetEntry(name="preview")
        time_without_hand = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30
            timestamp_ms += fps
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            hand_landmarker_result = hands_landmarker.detect_for_video(mp_frame, timestamp_ms)
            face_landmarker_result = face_landmarker.detect_for_video(mp_frame, timestamp_ms)
            frame = draw_landmarks_on_image(frame, hand_landmarker_result)
            if hand_landmarker_result.hand_landmarks:# and face_landmarker_result.face_landmarks:
                time_without_hand = 0
                gesture_frame = create_gesture_frame(
                    hand_landmarker_result, face_landmarker_result.face_landmarks[0] if face_landmarker_result.face_landmarks else None,
                    left=True,
                    right=True
                )
                gesture_data.add_frame(gesture_frame)

                left_sin = process_hand_orientation([gesture_frame.LEFT])[0]
                right_sin = process_hand_orientation([gesture_frame.RIGHT])[0]
                # print(right_sin)
                cv2.putText(frame, f"L: {left_sin[0]:.2f}",
                            (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                cv2.putText(frame, f"R: {right_sin[0]:.2f}",
                            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            else:
                time_without_hand += (1.0 / fps)


            if time_without_hand >= 1:
                if len(gesture_data.frames) > 3:
                    # url = "http://192.168.50.103:8090/api/classify_gesture/"
                    url = "http://signvision.arthurvanremoortel.me/api/classify_gesture/"
                    #url = "http://localhost:8000/api/classify_gesture/"
                    data = {
                        "language": "VGT",
                        "gesture": gesture_data.to_json(),
                    }
                    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
                    if response.status_code == 200:
                        print('Request successful')
                        print('Response content:', response.text)
                    else:
                        print('Request failed')
                        print('Response status code:', response.status_code)
                        print('Response content:', response.text)
                time_without_hand = 0
                gesture_data = GestureDatasetEntry(name="preview")

            # frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("MediaPipe Hands", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                stop = True
                break

    cap.release()