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