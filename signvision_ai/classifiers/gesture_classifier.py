import time

import numpy as np
from django.db.models import Prefetch
from tqdm import tqdm

from signvision_ai.models.gesture_model import GestureModel
from signvision_core.models import Country, SignLanguage, Word, Gesture, Hands
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame, GestureDataset, \
    process_hand_orientation, replace_nulls, process_hand_openness, distance_to_mouth, format_entry_task

MIN_GESTURE_DATASET_SIZE = 5


class GestureClassifier:
    def __init__(self, language: SignLanguage):
        self.sign_language: SignLanguage = language
        self.lookup_dict: {int: Gesture} = {}
        self.reverse_lookup_dict: {Gesture: int} = {}
        self.entry_lookup_dict: {str: int} = {}
        self.gesture_dataset: GestureDataset = GestureDataset([])  # Default to empty. Only load it when needed.
        self.gesture_model: GestureModel = GestureModel(self.gesture_dataset, self.sign_language)

        self.load_metadata()
        try:
            self.gesture_model.load_model()
        except Exception as e:
            print(f"Failed to load model for {self.sign_language.abbreviation}: {e}. Will train a new one instead.")
            self.load_dataset_entries()
            self.gesture_dataset.format_data(entry_lookup_dict=self.entry_lookup_dict)
            self.gesture_model.train_model(validate=False)
            self.gesture_model.save_model()

    def load_metadata(self) -> GestureDataset:
        entries = []
        # words = list(self.sign_language.words.prefetch_related(Prefetch('gestures__dataset')).all())
        words = list(self.sign_language.words.prefetch_related('gestures').all())
        gesture_id = 0
        for word in tqdm(words, desc=f"Loading dataset for {self.sign_language.abbreviation}", unit="words"):
            for gesture in word.gestures.all():
                # gesture_dataset = list(gesture.dataset.all())
                # if len(gesture_dataset) < MIN_GESTURE_DATASET_SIZE:
                #     print(f"Warning: Skipping {gesture} because it has {len(gesture_dataset)} entries. (<{MIN_GESTURE_DATASET_SIZE})")
                #     continue
                self.lookup_dict[gesture_id] = gesture
                self.reverse_lookup_dict[gesture] = gesture_id
                gesture_id += 1
        dataset = GestureDataset(entries)
        return dataset

    def load_dataset_entries(self):
        gestures = Gesture.objects.filter(word__language=self.sign_language).prefetch_related(Prefetch('dataset')).all()
        for gesture in gestures:
            gesture_id = self.reverse_lookup_dict[gesture]
            for dataset_entry in gesture.dataset.all():
                gesture_dataset_entry: GestureDatasetEntry = dataset_entry.to_gesture_dataset_entry()
                # gesture_dataset_entry.prepare()
                self.gesture_dataset.entries.append(gesture_dataset_entry)
                self.entry_lookup_dict[gesture_dataset_entry.name] = gesture_id

    def classify(self, entry: GestureDatasetEntry):
        entry, x_data = format_entry_task(entry)
        prediction = self.gesture_model.model.predict([x_data])
        predicted_gestures = {}
        for gesture_id, prediction in enumerate(prediction[0]):
            prediction = int(prediction * 100)
            if prediction > 5:
                try:
                    predicted_gestures[
                        self.lookup_dict[gesture_id]
                    ] = prediction
                except KeyError:
                    print(
                        f"Warning: {gesture_id} is present in the model but was probably deleted from the dataset. Retrain the model."
                    )
        return predicted_gestures
