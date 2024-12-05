from xgboost import callback
from typing import Any
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import DMatrix

import numpy as np
import pandas as pd



class XgbCallback(callback.TrainingCallback):
    def __init__(self, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.d_matrix = DMatrix(data=x_val, label=y_val)



class F1ScoreCallback(XgbCallback):
    def __init__(self, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        super().__init__(x_val, y_val)
        self.f1_scores = []


    def after_iteration(self, model: Any, epoch: int, evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]]) -> bool:
        y_pred_probs = model.predict(self.d_matrix, iteration_range=(0, epoch + 1))
        y_pred = np.argmax(y_pred_probs, axis=1)

        f1 = f1_score(self.y_val, y_pred, average='weighted')
        self.f1_scores.append(f1)

        print(f"Epoch: {epoch}, F1-Score: {f1:.4f}")

        return False


    def get_scores(self) -> list[float]:
        return self.f1_scores



class HistoryCallback(XgbCallback):
    def __init__(self, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        super().__init__(x_val, y_val)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }


    def after_iteration(self, model: Any, epoch: int,
                        evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]]) -> bool:

        train_loss = evals_log['train']['mlogloss'][-1] if 'train' in evals_log and 'mlogloss' in evals_log['train'] else None
        val_loss = evals_log['validation_0']['mlogloss'][-1] if 'validation_0' in evals_log and 'mlogloss' in evals_log['validation_0'] else None

        # Validation accuracy
        y_pred_probs = model.predict(self.d_matrix, iteration_range=(0, epoch + 1))
        y_pred = np.argmax(y_pred_probs, axis=1)

        val_acc = accuracy_score(self.y_val, y_pred)

        # Training accuracy (opcjonalnie, jeśli mamy dostęp do danych treningowych)
        if 'train' in evals_log and 'accuracy' in evals_log['train']:
            train_acc = evals_log['train']['accuracy'][-1]
        else:
            train_acc = None  # Brak danych o treningu w evals_log

        # Aktualizacja historii
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

        print(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return False


    def get_history(self) -> dict:
        return self.history

