import os
import time
from pathlib import Path
import cv2
from django.db.models import Q
from tqdm import tqdm
import numpy as np
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SignVisionSite.settings")
django.setup()

from signvision_core.models import Gesture, GestureEntry, SignLanguage
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame

window_width = 800
window_height = 600

VGT = SignLanguage.objects.get(abbreviation="VGT")

square_entry = GestureDatasetEntry("None")
square_entry.frames = [
    GestureFrame(
        LEFT=HandFrame(*[Coordinate(0.2, 0.2, 0.2) for _ in range(HandFrame.__annotations__.__len__())]),
        RIGHT=HandFrame(*[Coordinate.empty() for _ in range(HandFrame.__annotations__.__len__())]),
        MOUTH=Coordinate(0.5, 0.5, 0.5)
    ),
    GestureFrame(
        LEFT=HandFrame(*[Coordinate(0.2, 0.8, 0.2) for _ in range(HandFrame.__annotations__.__len__())]),
        RIGHT=HandFrame(*[Coordinate.empty() for _ in range(HandFrame.__annotations__.__len__())]),
        MOUTH=Coordinate(0.5, 0.5, 0.5)
    ),
    GestureFrame(
        LEFT=HandFrame(*[Coordinate(0.8, 0.8, 0.2) for _ in range(HandFrame.__annotations__.__len__())]),
        RIGHT=HandFrame(*[Coordinate.empty() for _ in range(HandFrame.__annotations__.__len__())]),
        MOUTH=Coordinate(0.5, 0.5, 0.5)
    ),
    GestureFrame(
        LEFT=HandFrame(*[Coordinate(0.8, 0.2, 0.2) for _ in range(HandFrame.__annotations__.__len__())]),
        RIGHT=HandFrame(*[Coordinate.empty() for _ in range(HandFrame.__annotations__.__len__())]),
        MOUTH=Coordinate(0.5, 0.5, 0.5)
    ),
    GestureFrame(
        LEFT=HandFrame(*[Coordinate(0.2, 0.2, 0.2) for _ in range(HandFrame.__annotations__.__len__())]),
        RIGHT=HandFrame(*[Coordinate.empty() for _ in range(HandFrame.__annotations__.__len__())]),
        MOUTH=Coordinate(0.5, 0.5, 0.5)
    ),
]
gestures = [None]
gestures = Gesture.objects.filter(Q(word__language=VGT) & Q(word__word__icontains="hallo")).prefetch_related('dataset').all()


def preview_entry(gesture_dataset_entry: GestureDatasetEntry):
    do_exit = False
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
                cv2.circle(image, center, 10 if 'TIP' not in landmark_name else 4, color,
                           -1)  # Draw a red circle at the coordinate

            for p1, p2 in [(hand_frame.WRIST, hand_frame.THUMB_BASE),
                           (hand_frame.WRIST, hand_frame.PINKY_BASE)]:
                cv2.line(image, (int(p1.x * window_width), int(p1.y * window_height)),
                         (int(p2.x * window_width), int(p2.y * window_height)), color, 1)

            fingers = [(hand_frame.THUMB_BASE, hand_frame.THUMB_TIP),
                       (hand_frame.INDEX_BASE, hand_frame.INDEX_TIP),
                       (hand_frame.MIDDLE_BASE, hand_frame.MIDDLE_TIP),
                       (hand_frame.RING_BASE, hand_frame.RING_TIP),
                       (hand_frame.PINKY_BASE, hand_frame.PINKY_TIP)]
            fi: int
            for fi, (fb, ft) in enumerate(fingers):
                if fi < len(fingers) - 1:
                    next_finger_base = fingers[fi + 1][0]
                    cv2.line(image, (int(fb.x * window_width), int(fb.y * window_height)),
                             (int(next_finger_base.x * window_width), int(next_finger_base.y * window_height)),
                             color, 1)

                cv2.line(image, (int(fb.x * window_width), int(fb.y * window_height)),
                         (int(ft.x * window_width), int(ft.y * window_height)), color, 1)

        if not frame.MOUTH.is_empty():
            cv2.circle(image, (int(frame.MOUTH.x * window_width), int(frame.MOUTH.y * window_height)), 20,
                       (255, 0, 0), -1)  # Draw a red circle at the coordinate

        cv2.imshow('Gesture Preview', image)
        if cv2.waitKey(5) & 0xFF == 27:
            do_exit = True
            break
        time.sleep(1.0 / 30.0)
    return do_exit


if __name__ == '__main__':
    do_exit = False
    while not do_exit:
        for gesture in gestures:
            gesture_entry: GestureEntry = gesture.dataset.all().first()
            # gesture_dataset_entry: GestureDatasetEntry = square_entry  # gesture_entry.to_gesture_dataset_entry()
            gesture_dataset_entry: GestureDatasetEntry = gesture_entry.to_gesture_dataset_entry()
            gesture_dataset_entry.prepare()
            do_exit = preview_entry(gesture_dataset_entry)
        if do_exit:
            break

