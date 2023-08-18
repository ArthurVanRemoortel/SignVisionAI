import time

from django.db.models import Prefetch
from tqdm import tqdm

from signvision_ai.models.gesture_model import GestureModel
from signvision_core.models import Country, SignLanguage, Word, Gesture, Hands
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame, GestureDataset

MIN_GESTURE_DATASET_SIZE = 5


class GestureClassifier:
    def __init__(self, language: SignLanguage):
        self.sign_language: SignLanguage = language
        self.lookup_dict: {int: Gesture} = {}
        self.reverse_lookup_dict: {Gesture: int} = {}
        self.entry_lookup_dict: {str: int} = {}
        self.gesture_dataset: GestureDataset = self.load_dataset()
        self.gesture_model: GestureModel = GestureModel(self.gesture_dataset, self.sign_language)

        try:
            self.gesture_model.load_model()
        except Exception as e:
            print(f"Failed to load model for {self.sign_language.abbreviation}: {e}. Will train a new one instead.")
            self.gesture_dataset.format_data(entry_lookup_dict=self.entry_lookup_dict)
            self.gesture_model.train_model()
            self.gesture_model.save_model()

    def load_dataset(self) -> GestureDataset:
        entries = []
        words = list(self.sign_language.words.prefetch_related(Prefetch('gestures__dataset')).all())
        gesture_id = 0
        for word in tqdm(words, desc=f"Loading dataset for {self.sign_language.abbreviation}", unit="words"):
            for gesture in word.gestures.all():
                gesture_dataset = list(gesture.dataset.all())
                if len(gesture_dataset) < MIN_GESTURE_DATASET_SIZE:
                    print(f"Warning: Skipping {gesture} because it has {len(gesture_dataset)} entries. (<{MIN_GESTURE_DATASET_SIZE})")
                    continue

                self.lookup_dict[gesture_id] = gesture
                self.reverse_lookup_dict[gesture] = gesture_id

                for entry in gesture_dataset:
                    gesture_dataset_entry: GestureDatasetEntry = entry.to_gesture_dataset_entry()
                    # gesture_dataset_entry.prepare()
                    entries.append(gesture_dataset_entry)
                    self.entry_lookup_dict[gesture_dataset_entry.name] = gesture_id
                gesture_id += 1
        dataset = GestureDataset(entries)
        # t0 = time.time()
        # dataset.format_data(entry_lookup_dict)
        # print(f"Formatted data in {time.time() - t0} seconds")
        return dataset

    def classify(self):
        ...
