import pandas as pd

import tensorflow as tf
import numpy as np

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.keras.callbacks import History
from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model import Model
from src.callbacks.F1ScoreCallback import F1ScoreCallbackBert



MAIN_DIR = "/home/loppiko/Documents/studies/5-sem/NLP/project/src/"


class BertModel(Model):

    tokenizer: PreTrainedTokenizerFast
    history: History
    cm: list[list[float]] | None = None
    f1_scores: list[float] = []

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

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        f1_callback = F1ScoreCallbackBert(val_dataset)

        if class_weights_dict:
            self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=3, class_weight=class_weights_dict, callbacks=[f1_callback])
        else:
            self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=3, callbacks=[f1_callback, early_stopping])

        self.f1_scores = f1_callback.f1_scores


    def predict(self, to_predict: list[list[str]]) -> np.ndarray:
        tokenized_data = self.tokenize_data(to_predict, max_length=500)
        test_dataset = tf.data.Dataset.from_tensor_slices(dict(tokenized_data)).batch(8)
        logits = self.model.predict(test_dataset).logits

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

        plt.figure(figsize=(18, 6))
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

        plt.subplot(1, 3, 3)
        plt.plot(self.f1_scores, label='Validation F1-score')
        plt.title('F1-score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{MAIN_DIR}saved-results/bert/{self.model_name}-accuracy.png")

        if self.cm is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['legal', 'spam', 'phishing', 'fraud'],
                        yticklabels=['legal', 'spam', 'phishing', 'fraud'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(f"{MAIN_DIR}saved-results/bert/{self.model_name}-confusion-matrix.png")
        else:
            print("Cannot save confusion-matrix, because it was not previously created")

        with open(f"{MAIN_DIR}saved-results/lstm/v2/data/{self.model_name}-final-metrics.txt", "w") as file:
            file.write(f"Accuracy: {history['accuracy'][-1]}\n")
            file.write(f"F1-Score: {self.f1_scores[-1]}\n")

    def train_bert_model(self, train_data: pd.DataFrame, train_labels: pd.DataFrame, max_length: int) -> None:
        # Podział danych na zestawy treningowe, walidacyjne i testowe
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            train_data.values.tolist(),
            train_labels.values.tolist(),
            test_size=0.3,
            random_state=28
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=28
        )

        # Oversampling dla danych treningowych
        oversampler = RandomOverSampler(random_state=28)
        X_train_resampled, Y_train_resampled = oversampler.fit_resample(np.array(X_train).reshape(-1, 1), Y_train)
        X_val_resampled, Y_val_resampled = oversampler.fit_resample(np.array(X_val).reshape(-1, 1), Y_val)

        X_train_resampled = X_train_resampled.flatten().tolist()
        X_val_resampled = X_val_resampled.flatten().tolist()

        X_test_df: list[list[str]] = X_test
        X_train, X_val, X_test = (
            self.tokenize_data(X_train_resampled, max_length),
            self.tokenize_data(X_val, max_length),
            self.tokenize_data(X_test, max_length)
        )

        # Tworzenie zestawów danych w TensorFlow
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), Y_train_resampled)).batch(8)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(X_val), Y_val)).batch(8)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), Y_test)).batch(8)

        train_counter = Counter(Y_train_resampled)
        val_counter = Counter(Y_val_resampled)

        print(train_counter)
        print(val_counter)

        print("X_train_resampled shape:", np.array(X_train_resampled).shape)
        print("Y_train_resampled shape:", np.array(Y_train_resampled).shape)

        print("X_val_resampled shape:", np.array(X_val).shape)
        print("Y_val_resampled shape:", np.array(Y_val).shape)

        print("X_test_resampled shape:", np.array(X_test).shape)
        print("Y_test_resampled shape:", np.array(Y_test).shape)


        # Obliczenie wag klas po oversamplingu
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.unique(Y_train_resampled),
        #     y=Y_train_resampled
        # )
        # class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        #
        # # Trenowanie modelu
        # self.fit(train_dataset, val_dataset, class_weights_dict)
        #
        # # Ewaluacja modelu
        # self.evaluate(test_dataset)
        # self.create_confusion_matrix(X_test_df, np.array(Y_test))
        # self.make_plots()



        # Obliczenie wag klas po oversamplingu
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.unique(Y_train_resampled),
        #     y=Y_train_resampled
        # )
        # class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        #
        # # Trenowanie modelu
        # self.fit(train_dataset, val_dataset, class_weights_dict)
        #
        # # Ewaluacja modelu
        # self.evaluate(test_dataset)
        # self.create_confusion_matrix(X_test_df, np.array(Y_test))
        # self.make_plots()


MODEL_TYPE = "bert-base-uncased"
MODEL_NAME = "bert-basic"

data = pd.read_csv(f"{MAIN_DIR}dataset/processed/final.csv").sample(n=1000, random_state=28)
body = data['body']
labels = data['label']
max_length = 500

bert_model = BertModel(MODEL_TYPE, MODEL_NAME)
bert_model.train_bert_model(body, labels, 500)

