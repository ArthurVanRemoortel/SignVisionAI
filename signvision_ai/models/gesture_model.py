from pathlib import Path

import keras
from sklearn.model_selection import train_test_split
from keras import backend as K
from signvision_core.models import Country, SignLanguage, Word, Gesture, Hands
from signvision_ai.dataset import GestureDatasetEntry, GestureFrame, Coordinate, HandFrame, GestureDataset
from keras.models import load_model, save_model
from keras.src.regularizers import L2
import tensorflow as tf


MODELS_ROOT = Path("data/saved_models")


class PrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, name='precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, 'bool')
        y_pred = K.cast(y_pred > 0.5, 'bool')

        true_positives = K.sum(K.cast(y_true & y_pred, 'float32'))
        false_positives = K.sum(K.cast(~y_true & y_pred, 'float32'))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, 'bool')
        y_pred = K.cast(y_pred > 0.5, 'bool')

        true_positives = K.sum(K.cast(y_true & y_pred, 'float32'))
        false_positives = K.sum(K.cast(~y_true & y_pred, 'float32'))
        false_negatives = K.sum(K.cast(y_true & ~y_pred, 'float32'))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


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
                    kernel_regularizer=L2(0.01),
                ),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.LSTM(
                    64, return_sequences=False, kernel_regularizer=L2(0.01)
                ),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",  # Experiment using different loss and metric functions.
            metrics=["sparse_categorical_accuracy"],
        )
        es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1, monitor='val_loss' if validate else 'loss')
        train_history = model.fit(
            x_train,
            y_train,
            epochs=500,
            batch_size=1,
            validation_data=(x_validate, y_validate) if validate else None,
            callbacks=[es_callback],
        )
        [loss, acc] = model.evaluate(x_test, y_test, verbose=1)
        print("Accuracy metric::" + str(acc))
        print("Loss:" + str(loss))
        self.model = model
        self._last_train_history = train_history

    def load_model(self):
        if not self.model_path.exists():
            raise Exception(f"Model file {self.model_path} does not exist.")
        self.model = load_model(self.model_path)

    def save_model(self):
        save_model(self.model, self.model_path)

    @property
    def model_path(self) -> Path:
        return MODELS_ROOT / (self.language.abbreviation + '-gesture.h5')
