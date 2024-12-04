from abc import ABC, abstractmethod
import pandas as pd

class Model(ABC):
    @abstractmethod
    def fit(self, *args) -> None:
        pass

    @abstractmethod
    def evaluate(self, *args) -> float:
        pass

    @abstractmethod
    def make_plots(self, *args) -> None:
        pass

    @abstractmethod
    def create_confusion_matrix(self, *args) -> None:
        pass