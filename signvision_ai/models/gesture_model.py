from pathlib import Path

import keras
from sklearn.model_selection import train_test_split

from signvision_core.models import Country, SignLanguage, Word, Gesture, Hands
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame, GestureDataset
from keras.models import load_model, save_model
from keras.src.regularizers import L2
import tensorflow as tf


MODELS_ROOT = Path("data/saved_models")


class GestureModel:
    def __init__(self, dataset: GestureDataset, language: SignLanguage):
        self.language: SignLanguage = language
        self.dataset: GestureDataset = dataset
        self.model = None
        self._last_train_history = None

    def train_model(self, train_size=0.75, validate=False):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.x_data,
            self.dataset.y_data,
            train_size=train_size,
            random_state=42,
            stratify=self.dataset.y_data,
            shuffle=True,
        )
        x_validate = []
        y_validate = []
        if validate:
            x_validate, x_test, y_validate, y_test = train_test_split(
                x_test,
                y_test,
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=y_test,
            )

        num_classes = len(set(self.dataset.y_data))

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(
                    32,
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                    return_sequences=True,
                    kernel_regularizer=L2(0.001),
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(
                    64, return_sequences=False, kernel_regularizer=L2(0.001)
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",  # Experiment using different loss and metric functions.
            metrics=["sparse_categorical_accuracy"],
        )

        es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1, monitor='loss')
        train_history = model.fit(
            x_train,
            y_train,
            epochs=1000,
            batch_size=1,
            validation_data=(x_validate, y_validate) if validate else None,
            callbacks=[es_callback],
        )
        [loss, acc] = model.evaluate(x_test, y_test, verbose=1)
        print("Accuracy:" + str(acc))
        print("Loss:" + str(loss))
        self.model = model
        self._last_train_history = train_history

    def load_model(self):
        if not self.model_path.exists():
            raise Exception(f"Model file {self.model_path} does not exist.")
        self.model = load_model(self.model_path)

    def save_model(self):
        save_model(self.model, self.model_path)

    def classify(self):
        ...

    @property
    def model_path(self) -> Path:
        return MODELS_ROOT / (self.language.abbreviation + '-gesture.keras')
