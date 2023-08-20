from enum import Enum, IntEnum

from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.utils.translation import gettext_lazy as _

import signvision_ai.dataset as dataset


class Hands(models.TextChoices):
    LEFT = 'left'
    RIGHT = 'right'
    BOTH = 'both'


class GestureEntryType(models.TextChoices):
    MEDIAPIPE_VIDEO = 'mediapipe_video'
    UNKNOWN = 'unknown'


class Country(models.Model):
    name = models.CharField(max_length=100, null=False)
    sign_languages = models.ManyToManyField('SignLanguage', related_name="countries")

    class Meta:
        db_table = "countries"

    def __str__(self):
        return self.name


class SignLanguage(models.Model):
    name = models.CharField(max_length=100, null=False)
    abbreviation = models.CharField(max_length=100, null=True)
    description = models.TextField(null=True)

    class Meta:
        db_table = "sign_languages"

    def __str__(self):
        return self.name


class Word(models.Model):
    word = models.CharField(max_length=255, null=False)
    language = models.ForeignKey(SignLanguage, on_delete=models.CASCADE, related_name='words', null=False)

    class Meta:
        db_table = "words"

    def __str__(self) -> str:
        return self.word


class Gesture(models.Model):
    word = models.ForeignKey(Word, on_delete=models.CASCADE, related_name='gestures', null=False)
    hands = models.CharField(choices=Hands.choices, default=Hands.BOTH, null=False)

    class Meta:
        db_table = "gestures"

    @property
    def token(self):
        return f"{self.word}:{self.id}"

    def __str__(self):
        return f"{self.word.word} ({self.hands})"


class GestureEntry(models.Model):
    gesture = models.ForeignKey(Gesture, on_delete=models.CASCADE, related_name='dataset', null=False)
    source_name = models.CharField(max_length=255, null=True)
    frame_count = models.IntegerField(null=False)
    entry_type = models.CharField(choices=GestureEntryType.choices, default=GestureEntryType.UNKNOWN, null=False)
    LEFT_WRIST = ArrayField(models.JSONField())
    LEFT_THUMB_BASE = ArrayField(models.JSONField())
    LEFT_THUMB_TIP = ArrayField(models.JSONField())
    LEFT_INDEX_BASE = ArrayField(models.JSONField())
    LEFT_INDEX_TIP = ArrayField(models.JSONField())
    LEFT_MIDDLE_BASE = ArrayField(models.JSONField())
    LEFT_MIDDLE_TIP = ArrayField(models.JSONField())
    LEFT_RING_BASE = ArrayField(models.JSONField())
    LEFT_RING_TIP = ArrayField(models.JSONField())
    LEFT_PINKY_BASE = ArrayField(models.JSONField())
    LEFT_PINKY_TIP = ArrayField(models.JSONField())

    RIGHT_WRIST = ArrayField(models.JSONField())
    RIGHT_THUMB_BASE = ArrayField(models.JSONField())
    RIGHT_THUMB_TIP = ArrayField(models.JSONField())
    RIGHT_INDEX_BASE = ArrayField(models.JSONField())
    RIGHT_INDEX_TIP = ArrayField(models.JSONField())
    RIGHT_MIDDLE_BASE = ArrayField(models.JSONField())
    RIGHT_MIDDLE_TIP = ArrayField(models.JSONField())
    RIGHT_RING_BASE = ArrayField(models.JSONField())
    RIGHT_RING_TIP = ArrayField(models.JSONField())
    RIGHT_PINKY_BASE = ArrayField(models.JSONField())
    RIGHT_PINKY_TIP = ArrayField(models.JSONField())

    MOUTH = ArrayField(models.JSONField())

    class Meta:
        db_table = "gesture_entries"

    def to_gesture_dataset_entry(self) -> dataset.GestureDatasetEntry:
        entry = dataset.GestureDatasetEntry(name=self.source_name)
        for frame_n in range(len(self.MOUTH)):
            left_landmarks_json = [getattr(self, "LEFT_"+field)[frame_n] for field in dataset.HandFrame.__annotations__.keys()]
            right_landmarks_json = [getattr(self, "RIGHT_"+field)[frame_n] for field in dataset.HandFrame.__annotations__.keys()]
            left_landmarks = [dataset.Coordinate(**landmark) for landmark in left_landmarks_json]
            right_landmarks = [dataset.Coordinate(**landmark) for landmark in right_landmarks_json]
            entry.add_frame(dataset.GestureFrame(dataset.HandFrame(*left_landmarks), dataset.HandFrame(*right_landmarks), dataset.Coordinate(**self.MOUTH[frame_n])))
        return entry

    @classmethod
    def from_gesture_dataset_entry(cls, gesture: Gesture, entry: dataset.GestureDatasetEntry, source_name: str = None) -> 'GestureEntry':
        new_entry = GestureEntry(gesture=gesture, source_name=source_name, frame_count=len(entry.frames))
        mouth_values = []
        for field in dataset.HandFrame.__annotations__.keys():
            field_values_left = []
            field_values_right = []
            for frame in entry.frames:
                frame_field_coordinate_left: dataset.Coordinate = frame.LEFT.__getattribute__(field)
                frame_field_coordinate_right: dataset.Coordinate = frame.RIGHT.__getattribute__(field)
                field_values_left.append(frame_field_coordinate_left.to_dict())
                field_values_right.append(frame_field_coordinate_right.to_dict())
                # mouth_values.append(frame.MOUTH.to_dict())
            setattr(new_entry, "LEFT_"+field, field_values_left)
            setattr(new_entry, "RIGHT_"+field, field_values_right)

        for frame in entry.frames:
            mouth_values.append(frame.MOUTH.to_dict())
        new_entry.MOUTH = mouth_values
        return new_entry




