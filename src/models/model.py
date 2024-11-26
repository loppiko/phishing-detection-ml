from abc import ABC, abstractmethod
import pandas as pd

class Model(ABC):

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def evaluate(self, x_val: pd.DataFrame, y_val: pd.DataFrame) -> float:
        pass