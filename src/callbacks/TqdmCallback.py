from tqdm import tqdm
from xgboost import callback, Booster


class TqdmCallback(callback.TrainingCallback):
    def __init__(self) -> None:
        super().__init__()
        self.tqdm_bar = None

    def before_training(self, model: Booster) -> None:
        self.tqdm_bar = tqdm(total=model.attributes().get('n_estimators'), desc="Training Progress")

    def after_iteration(self, model: Booster, epoch, evals_log) -> bool:
        self.tqdm_bar.update(1)
        return False

    def after_training(self, model: Booster) -> None:
        self.tqdm_bar.close()