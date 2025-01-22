from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
from sklearn.metrics import f1_score


class F1ScoreCallback(Callback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.x_val), axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.f1_scores.append(f1)



class F1ScoreCallbackBert(Callback):
    def __init__(self, val_dataset):
        val_data = list(val_dataset.as_numpy_iterator())
        self.x_val = {key: np.stack([item[0][key] for item in val_data]) for key in val_data[0][0].keys()}
        self.y_val = np.array([item[1] for item in val_data])
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.x_val).logits, axis=1)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        self.f1_scores.append(f1)
