from abc import ABC, abstractmethod
import pandas as pd

class Model(ABC):
    @abstractmethod
    def fit(self, *args) -> None:
        pass

    @abstractmethod
    def evaluate(self, *args) -> float:
        pass
