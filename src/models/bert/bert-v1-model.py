from platform import win32_edition

import pandas as pd

import tensorflow as tf
import numpy as np

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.keras.callbacks import History

import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model import Model



MAIN_DIR = "/content/model/"


class BertModel(Model):

    tokenizer: PreTrainedTokenizerFast
    history: History
    cm: list[list[float]] | None = None

    def __init__(self, model_type: str, model_name: str) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_type, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)


    def fit(self, train_dataset: DatasetV2, val_dataset: DatasetV2, class_weights_dict: dict | None = None) -> None:
        self.model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        if class_weights_dict:
            self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=3, class_weight=class_weights_dict)
        else:
            self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=3)


    def predict(self, to_predict: list[list[str]]) -> np.ndarray:
        tokenized_data = self.tokenize_data(to_predict, max_length=500)
        test_dataset = tf.data.Dataset.from_tensor_slices(dict(tokenized_data)).batch(8)
        logits = self.model.predict(test_dataset).logits

        # Konwersja logitów na etykiety (największa wartość logitu dla każdej próbki)
        return tf.argmax(logits, axis=1).numpy()


    def evaluate(self, test_dataset: DatasetV2) -> None:
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"loss: {loss}, accuracy: {accuracy}")


    def create_confusion_matrix(self, x_test: list[list[str]], y_test: np.ndarray) -> None:
        y_pred = self.predict(x_test)
        self.cm = confusion_matrix(y_test, y_pred)


    def tokenize_data(self, data: list[list[str]], max_length: int):
        return self.tokenizer(
            data,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

    def save_model(self) -> None:
        self.model.save_pretrained(f"{MAIN_DIR}models/bert/saved/{self.model_name}-model.keras")
        self.tokenizer.save_pretrained(f"{MAIN_DIR}{self.model_name}-tokenizer.keras")
        print(f"Model saved to {MAIN_DIR}models/bert/saved/")


    def make_plots(self) -> None:
        history = self.history.history

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{MAIN_DIR}saved-results/bert/{self.model_name}-accuracy.png")

        if self.cm is not None:
            plt.figure(figsize=(8, 6))
            labels = list(self.model.config.id2label.values()) if hasattr(self.model.config, "id2label") else None
            sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(f"{MAIN_DIR}saved-results/bert/{self.model_name}-confusion-matrix.png")
        else:
            print("Cannot save confusion-matrix, because it was not previously created")


    def train_bert_model(self, train_data: pd.DataFrame, train_labels: pd.DataFrame, max_length: int) -> None:
        X_train, X_temp, Y_train, Y_temp = train_test_split(train_data.values.tolist(), train_labels.values.tolist(), test_size=0.3, random_state=28)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

        X_test_df: list[list[str]] = X_test

        X_train, X_val, X_test = self.tokenize_data(X_train, max_length), self.tokenize_data(X_val, max_length), self.tokenize_data(X_test, max_length)

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), Y_train)).batch(8)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(X_val), Y_val)).batch(8)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), Y_test)).batch(8)


        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.values), y=labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        self.fit(train_dataset, val_dataset, class_weights_dict)
        self.evaluate(test_dataset)
        self.create_confusion_matrix(X_test_df, np.array(Y_test))
        self.make_plots()


MODEL_TYPE = "bert-base-uncased"
MODEL_NAME = "bert-basic"

data = pd.read_csv(f"{MAIN_DIR}dataset/processed/final.csv").sample(n=1000, random_state=28)
body = data['body']
labels = data['label']
max_length = 500

bert_model = BertModel(MODEL_TYPE, MODEL_NAME)
bert_model.train_bert_model(body, labels, 500)

