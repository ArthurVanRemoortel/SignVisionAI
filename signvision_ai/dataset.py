import copy
import functools
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pprint import pprint
from typing import Dict, List
import numpy as np
from tqdm import tqdm


def format_entry_task(entry: 'GestureDatasetEntry') -> ('GestureDatasetEntry', list[[float]]):
    entry.prepare()
    left_hand_orientations = process_hand_orientation([frame.LEFT for frame in entry.frames])
    right_hand_orientations = process_hand_orientation([frame.RIGHT for frame in entry.frames])
    entry.frames = distance_to_mouth(entry.frames)
    x_data = [
        replace_nulls([frame.LEFT.WRIST.z for frame in entry.frames]),
        replace_nulls([frame.RIGHT.WRIST.z for frame in entry.frames]),
        replace_nulls([frame.LEFT.WRIST.x for frame in entry.frames]),
        replace_nulls([frame.LEFT.WRIST.y for frame in entry.frames]),
        replace_nulls([frame.RIGHT.WRIST.x for frame in entry.frames]),
        replace_nulls([frame.RIGHT.WRIST.y for frame in entry.frames]),
        replace_nulls([frame.LEFT.MIDDLE_TIP.x for frame in entry.frames]),
        replace_nulls([frame.LEFT.MIDDLE_TIP.y for frame in entry.frames]),
        replace_nulls([frame.RIGHT.MIDDLE_TIP.x for frame in entry.frames]),
        replace_nulls([frame.RIGHT.MIDDLE_TIP.y for frame in entry.frames]),
        *left_hand_orientations,
        *right_hand_orientations,
        *process_hand_openness([frame.LEFT for frame in entry.frames]),
        *process_hand_openness([frame.RIGHT for frame in entry.frames]),
    ]
    return entry, x_data


class GestureDataset:
    def __init__(self, gesture_entries: ['GestureDatasetEntry']):
        self.entries: ['GestureDatasetEntry'] = gesture_entries
        self.x_data = np.array([])
        self.y_data = np.array([])

    def format_data(self, entry_lookup_dict: {'GestureDatasetEntry': int}):
        x_dataset = []
        y_dataset = []

        with Pool(processes=2) as pool:
            entries_results_x_data = pool.imap_unordered(
                format_entry_task,
                self.entries,
            )
            entry: GestureDatasetEntry
            for (entry, x_data) in entries_results_x_data:
                y_dataset.append(entry_lookup_dict[entry.name])
                x_dataset.append(x_data)

        self.y_data = np.array(y_dataset)
        self.x_data = np.array(x_dataset, dtype='float')


@dataclass
class Coordinate:
    x: float | None
    y: float | None
    z: float | None

    @classmethod
    def from_dict(cls, data: dict) -> 'Coordinate':
        return cls(data['x'], data['y'], data['z'])

    def to_dict(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @classmethod
    def empty(cls) -> 'Coordinate':
        return cls(None, None, None)

    def is_empty(self) -> bool:
        return self.x is None or self.y is None or self.z is None

    @classmethod
    def neutral(cls) -> 'Coordinate':
        return cls(-1, -1, -1)

    def merge_with(self, other: 'Coordinate') -> 'Coordinate':
        if self.is_empty():
            return other
        elif other.is_empty():
            return self
        else:
            return Coordinate((self.x + other.x) / 2, (self.y + other.y) / 2, (self.z + other.z) / 2)

    def to_json(self) -> {}:
        return self.to_dict()

    @classmethod
    def from_json(cls, json_data: {}) -> 'Coordinate':
        return cls.from_dict(json_data)

    def __add__(self, other: 'Coordinate') -> 'Coordinate':
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Coordinate') -> 'Coordinate':
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, value: float | int) -> 'Coordinate':
        return Coordinate(self.x / value, self.y / value, self.z / value)


@dataclass
class HandFrame:
    WRIST: Coordinate
    THUMB_BASE: Coordinate
    THUMB_TIP: Coordinate
    INDEX_BASE: Coordinate
    INDEX_TIP: Coordinate
    MIDDLE_BASE: Coordinate
    MIDDLE_TIP: Coordinate
    RING_BASE: Coordinate
    RING_TIP: Coordinate
    PINKY_BASE: Coordinate
    PINKY_TIP: Coordinate

    @classmethod
    def empty(cls) -> 'HandFrame':
        return cls(*[Coordinate.empty()]*len(HandFrame.__annotations__))

    @classmethod
    def neutral(cls) -> 'HandFrame':
        return cls(*[Coordinate.neutral()]*len(HandFrame.__annotations__))

    def is_empty(self) -> bool:
        for field in HandFrame.__annotations__.keys():
            value = getattr(self, field)
            if value.is_empty():
                return True  # NOTE: This means a hand frame where 1 landmark is empty is considered empty.
        return False

    def merge_with(self, other: 'HandFrame') -> 'HandFrame':
        landmarks: [Coordinate] = []
        for field in HandFrame.__annotations__.keys():
            value: Coordinate = getattr(self, field)
            other_value: Coordinate = getattr(other, field)
            if not value.is_empty() and not other_value.is_empty():
                new_value: Coordinate = value.merge_with(other_value)
            else:
                new_value: Coordinate = Coordinate.empty()
            landmarks.append(new_value)
        return HandFrame(*landmarks)

    def landmark_values(self) -> list[Coordinate]:
        return [getattr(self, field) for field in HandFrame.__annotations__.keys()]

    def landmark_values_iterator(self) -> list[Coordinate]:
        for field in HandFrame.__annotations__.keys():
            yield getattr(self, field)

    def to_json(self) -> {}:
        json_out = {}
        for field in HandFrame.__annotations__.keys():
            json_out[field] = getattr(self, field).to_dict()
        return json_out

    @classmethod
    def from_json(cls, json_data: {}) -> 'HandFrame':
        return cls(*[Coordinate.from_json(json_data[field]) for field in HandFrame.__annotations__.keys()])


@dataclass
class GestureFrame:
    LEFT: HandFrame
    RIGHT: HandFrame
    MOUTH: Coordinate

    _empty_instance = None

    @classmethod
    def empty(cls) -> 'GestureFrame':
        return cls(HandFrame.empty(), HandFrame.empty(), Coordinate.empty())

    def is_empty(self) -> bool:
        return self.LEFT.is_empty() and self.RIGHT.is_empty() and self.MOUTH.is_empty()

    def merge_with(self, other: 'GestureFrame') -> 'GestureFrame':
        new_left = self.LEFT.merge_with(other.LEFT)
        new_right = self.RIGHT.merge_with(other.RIGHT)
        new_mouth = self.MOUTH.merge_with(other.MOUTH)
        new_value = GestureFrame(
            LEFT=new_left,
            RIGHT=new_right,
            MOUTH=new_mouth,
        )
        return new_value

    def to_json(self):
        return {
            "LEFT": self.LEFT.to_json(),
            "RIGHT": self.RIGHT.to_json(),
            "MOUTH": self.MOUTH.to_json(),
        }

    @classmethod
    def from_json(cls, json_data: {}) -> 'GestureFrame':
        return cls(
            LEFT=HandFrame.from_json(json_data["LEFT"]),
            RIGHT=HandFrame.from_json(json_data["RIGHT"]),
            MOUTH=Coordinate.from_json(json_data["MOUTH"]),
        )


class GestureDatasetEntry:
    def __init__(self, name: None | str):
        self.frames: [GestureFrame] = []
        self.name = name

    def add_frame(self, frame: GestureFrame):
        self.frames.append(frame)

    def prepare(self, mouth_distance=True):
        self.frames = strip_empty_frames(self.frames)
        self.frames = normalize_length(self.frames, 100)
        self.frames = interpolate_empty_frames(self.frames)
        # if mouth_distance:
        #     self.frames = distance_to_mouth(self.frames)

    def populated_frames(self) -> list[GestureFrame]:
        empty_frame = GestureFrame.empty()
        return [frame for frame in self.frames if frame != empty_frame]

    def populated_rate(self) -> float:
        return len(self.populated_frames()) / len(self.frames)

    def to_json(self):
        return {frame_n: frame.to_json() for frame_n, frame in enumerate(self.frames)}

    @classmethod
    def from_json(cls, json_data: dict) -> 'GestureDatasetEntry':
        entry = cls(name=None)
        for frame_n, frame_json in json_data.items():
            mouth = frame_json["MOUTH"]
            entry.add_frame(GestureFrame.from_json(frame_json))
        return entry


def strip_empty_frames(frames: list[GestureFrame]) -> list[GestureFrame]:
    new_frames = []
    for frame in frames:
        if not frame.LEFT.WRIST.is_empty() or not frame.RIGHT.WRIST.is_empty():
            new_frames.append(frame)
    return new_frames


def normalize_length(frames: list[GestureFrame], length: int) -> list[GestureFrame]:
    new_frames = shrink_frames(extend_frames(frames, length), length)
    return new_frames


def distance_to_mouth(frames: list[GestureFrame], hand_frame=None):
    new_frames = []
    average_mouth_pos: Coordinate = Coordinate(0, 0, 0)
    count = 0
    for frame in frames:
        if frame.MOUTH and not frame.MOUTH.is_empty():
            count += 1
            average_mouth_pos += frame.MOUTH
    if count == 0:
        average_mouth_pos = Coordinate(0.5, 0.5, 0.5)
    else:
        average_mouth_pos /= count

    frame: GestureFrame
    for i, frame in enumerate(frames):
        new_left_hands = []
        new_right_hands = []
        for hand_name in ['LEFT', 'RIGHT']:
            hand_frame: HandFrame = frame.__getattribute__(hand_name)
            new_frame_values = []
            if hand_frame.is_empty():
                new_frame_values = [Coordinate.empty()] * len(HandFrame.__annotations__)
            else:
                for landmark_value in hand_frame.landmark_values_iterator():
                    new_frame_values.append(
                        Coordinate(
                            (landmark_value.x - average_mouth_pos.x),
                            (landmark_value.y - average_mouth_pos.y),
                            (landmark_value.z - average_mouth_pos.z),
                        )
                    )
            new_hand_frame = HandFrame(*new_frame_values)
            if hand_name == "LEFT":
                new_left_hands = new_hand_frame
            if hand_name == "RIGHT":
                new_right_hands = new_hand_frame

        new_frames.append(GestureFrame(new_left_hands, new_right_hands, frame.MOUTH))
    return new_frames


def interpolate_empty_frames(frames: list[GestureFrame]) -> list[GestureFrame]:
    new_frames = frames
    for hand_name in ['LEFT', 'RIGHT']:
        frame: GestureFrame
        for i, frame in enumerate(new_frames):
            hand_frame: HandFrame = frame.__getattribute__(hand_name)
            new_hand_frame: HandFrame = None
            if hand_frame.is_empty():
                left_neighbor = None
                right_neighbor = None
                # Find the nearest left neighbor
                for l in range(i - 1, -1, -1):
                    if not new_frames[l].__getattribute__(hand_name).is_empty():
                        left_neighbor = new_frames[l]
                        break
                # Find the nearest right neighbor
                for r in range(i + 1, len(new_frames)):
                    if not new_frames[r].__getattribute__(hand_name).is_empty():
                        right_neighbor = new_frames[r]
                        break
                # Impute the null value with the average of neighbors
                if left_neighbor is not None and right_neighbor is not None:
                    new_hand_frame = left_neighbor.__getattribute__(hand_name).merge_with(right_neighbor.__getattribute__(hand_name))
                if right_neighbor is None and left_neighbor:
                    new_hand_frame = left_neighbor.__getattribute__(hand_name)
                if not new_hand_frame or new_hand_frame.is_empty():
                    continue
                if hand_name == "LEFT":
                    new_frames[i].LEFT = new_hand_frame
                    new_frames[i].MOUTH = left_neighbor.MOUTH.merge_with(right_neighbor.MOUTH) if right_neighbor else left_neighbor.MOUTH
                elif hand_name == "RIGHT":
                    new_frames[i].RIGHT = new_hand_frame
                    new_frames[i].MOUTH = left_neighbor.MOUTH.merge_with(right_neighbor.MOUTH) if right_neighbor else left_neighbor.MOUTH

    return new_frames


def shrink_frames(frames: list[GestureFrame], length: int) -> list[GestureFrame]:
    if len(frames) <= length:
        return frames
    skip_step = length / len(frames)
    new_frames = [None for _ in range(length)]
    # grouped_frames = [[] for _ in range(length)]
    for i, frame in enumerate(frames):
        insert_index = int(i * skip_step)
        # grouped_frames[insert_index].append(frame)
        existing_at_index = new_frames[insert_index]
        if existing_at_index:
            new_frame = existing_at_index.merge_with(frame)
            new_frames[insert_index] = new_frame
        else:
            new_frames[insert_index] = frame
    # TODO: Actual average of overlapping frames. Currently biased toward the last frame.
    return new_frames


def extend_frames(frames: list[GestureFrame], length: int) -> list[GestureFrame]:
    if len(frames) >= length:
        return frames
    skip_step = length / len(frames)
    initial_length = len(frames)
    num_none_between_values = (length - initial_length) // (initial_length - 1)

    new_frames: [GestureFrame] = []
    for i, value in enumerate(frames):
        if i < initial_length - 1:
            new_frames.append(value)
            new_frames.extend([None] * num_none_between_values)
        else:
            new_frames.append(value)

    remaining_none_values = length - initial_length - (num_none_between_values * (initial_length - 1))
    new_frames.extend([None] * remaining_none_values)

    for i, frame in enumerate(new_frames):
        if frame is None:
            new_frames[i] = GestureFrame.empty()
    return new_frames


def scale_number(number, number_min, number_max, new_min=0, new_max=1):
    number_range = number_max - number_min
    new_range = new_max - new_min
    return (((number - number_min) * new_range) / number_range) + new_min


def calculate_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def calculate_angle(x0: float, y0: float, x1: float, y1: float) -> float:
    angle = math.atan2(y0 - y1, x0 - x1)
    return np.degrees(angle) % 360.0


def process_hand_orientation(hand_frames: [HandFrame]) -> [float]:
    angles = []
    prev_angle = 0
    hand: HandFrame
    for hand in hand_frames:
        base = hand.WRIST
        tip = hand.MIDDLE_TIP
        if base.is_empty():
            angles.append(prev_angle)
            continue
        angle = calculate_angle(base.x, base.y, tip.x, tip.y)
        angles.append(angle)
        prev_angle = angle
    angles_rad = np.radians(angles)  # Convert to radians
    sines = np.sin(angles_rad)
    cosines = np.cos(angles_rad)
    return [sines, cosines]


def process_hand_openness(hand_frames: [HandFrame]) -> [[float]]:
    openness = [[], [], []]
    for hand_frame in hand_frames:
        if hand_frame.WRIST.is_empty():
            for i, (finger_base, finger_tip) in enumerate(
                    [(hand_frame.INDEX_BASE, hand_frame.INDEX_TIP), (hand_frame.MIDDLE_BASE, hand_frame.MIDDLE_TIP),
                     (hand_frame.RING_BASE, hand_frame.RING_TIP)]
            ):
                openness[i].append(0.0)  # TODO: Last value for finger.
            continue
        palm_size = calculate_distance(hand_frame.WRIST.x, hand_frame.WRIST.y, hand_frame.MIDDLE_BASE.x, hand_frame.MIDDLE_BASE.y)
        hand_size = palm_size + calculate_distance(
            hand_frame.MIDDLE_TIP.x, hand_frame.MIDDLE_TIP.y, hand_frame.MIDDLE_TIP.x, hand_frame.MIDDLE_TIP.y
        )

        for i, (finger_base, finger_tip) in enumerate(
                [(hand_frame.INDEX_BASE, hand_frame.INDEX_TIP), (hand_frame.MIDDLE_BASE, hand_frame.MIDDLE_TIP),
                 (hand_frame.RING_BASE, hand_frame.RING_TIP)]
        ):
            # Check if tip is in palm are. Is inbetween wrist and finger base.
            if is_point_inside_triangle((finger_tip.x, finger_tip.y), (hand_frame.WRIST.x, hand_frame.WRIST.y), (hand_frame.THUMB_TIP.x, hand_frame.THUMB_TIP.y), (hand_frame.PINKY_BASE.x, hand_frame.PINKY_BASE.y)):
                openness[i].append(0.0)
            else:
                distance = calculate_distance(finger_base.x, finger_base.y, finger_tip.x, finger_tip.y)
                relative_distance = distance / hand_size
                scaled_distance = scale_number(relative_distance, 0.1, 0.9)
                scaled_distance = max(0.0, min(scaled_distance, 1.0))
                openness[i].append(scaled_distance)
            # openness[i].append(scaled_distance)
    # pprint(openness)
    return openness


def replace_nulls(values: [float | None]) -> [float]:
    return [-1 if value is None else value for value in values]


def is_point_inside_triangle(p, a, b, c):
    # Thanks, ChatGPT <3
    # Calculate barycentric coordinates
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])

    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denominator
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denominator
    gamma = 1 - alpha - beta

    # Check if the point is inside the triangle
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1
