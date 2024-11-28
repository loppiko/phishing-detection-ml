import pandas as pd

import tensorflow as tf

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from src.models.model import Model


class BertModel(Model):

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def fit(self, train_dataset: DatasetV2, val_dataset: DatasetV2) -> None:
        self.model.compile(
            optimizer=Adam(learning_rate=5e-5),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        self.model.fit(train_dataset, validation_data=val_dataset, epochs=3)


    def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
        ...


    def evaluate(self, test_dataset: DatasetV2) -> None:
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"loss: {loss}, accuracy: {accuracy}")


    def tokenize_data(self, data: list[list[str]], max_length: int):
        return self.tokenizer(
            data,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )


    def train_bert_model(self, train_data: pd.DataFrame, train_labels: pd.DataFrame, max_length: int) -> None:
        X_train, X_temp, Y_train, Y_temp = train_test_split(train_data.values.tolist(), train_labels.values.tolist(), test_size=0.3, random_state=28)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

        X_train, X_val, X_test = self.tokenize_data(X_train, max_length), self.tokenize_data(X_val, max_length), self.tokenize_data(X_test, max_length)

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), Y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(X_val), Y_val)).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), Y_test)).batch(32)

        self.fit(train_dataset, val_dataset)
        self.evaluate(test_dataset)



MODEL_NAME = "bert-base-uncased"

bert_model = BertModel(MODEL_NAME)
