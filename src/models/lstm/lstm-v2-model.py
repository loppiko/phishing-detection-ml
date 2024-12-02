from typing import Tuple

from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

from src.models.model import Model
from src.callbacks.F1ScoreCallback import F1ScoreCallback

# TODO
### 🔷 1. Compare model with and without stopwords
### 🔷 2. Use padding to cut additional part of the message (max_len) - Normalizacja długości sekwencji
### 🔷 3. Only domain
###     3a. Only domains
###    🔷 3b. Domain matches
### 🔷 4. LSTM Layers (return_sequences=True)
### 🔷 5. Dropout -> Zapobiega przeuczeniu
### ? Bidirectional
### 6. Hiper-parameters opt.
### 🔷 7. Early Stopping
### 🔷 8. Resolve unbalanced dataset problem
### 9. Compare all models
### ? Cross validation
### 🔷 ? F1-score -> Not eqauls dataset, Precision/Recall -> Fal-Pos, Pos-Fal have meaning


## 1 -> ONLY BODY
## 2 -> STOP WORDS BODY
## 3 -> DOMAIN
## 4 -> SUBJECT
## 5 -> FULL MODEL


MAIN_DIR = "/content/model/"


class LSTMModelAdvanced(Model):
    history = None
    cm: list[list[float]] = []
    f1_scores: list[float] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = Sequential([
            Embedding(input_dim=5000, output_dim=128),
            LSTM(128),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # Wyjście dla 4 klas
        ])


    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame):
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1)
        )

        class_weights_dict = dict(enumerate(class_weights))
        f1_callback = F1ScoreCallback(x_val, y_val)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train, epochs=40, batch_size=16, validation_split=0.2, callbacks=[early_stopping, f1_callback], class_weight=class_weights_dict)

        self.f1_scores = f1_callback.f1_scores


    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")


    def create_confusion_matrix(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred_probs = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        self.cm = confusion_matrix(y_true, y_pred)


    def make_plots(self) -> None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['legal', 'spam', 'phishing', 'fraud'],
                    yticklabels=['legal', 'spam', 'phishing', 'fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"{MAIN_DIR}saved-results/lstm/v2/plots/{self.model_name}-conf-matrix.png")
        plt.clf()

        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(18, 6))

        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # F1-score plot
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(self.f1_scores) + 1), self.f1_scores, label='Validation F1-Score', color='orange')
        plt.title('F1-Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{MAIN_DIR}saved-results/lstm/v2/plots/{self.model_name}-accuracy.png")
        plt.close()


        final_accuracy = val_accuracy[-1]
        final_f1_score = self.f1_scores[-1]
        with open(f"{MAIN_DIR}saved-results/lstm/v2/data/{self.model_name}-final-metrics.txt", "w") as file:
            file.write(f"Accuracy: {final_accuracy}\n")
            file.write(f"F1-Score: {final_f1_score}\n")


    def saving_model(self, save_path: str = "") -> None:
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            self.model.save(f"{MAIN_DIR}models/lstm/saved/{self.model_name}-model.keras")
            print(f"Model saved to {MAIN_DIR}models/lstm/saved/{self.model_name}-model.keras")


    @staticmethod
    def loading_model(load_path: str) -> Model:
        return load_model(load_path)


def train_lstm_model(lstm_model: LSTMModelAdvanced, train_data: np.ndarray, train_labels: np.ndarray, max_length: int, model_save_path: str = "") -> None:
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data)

    sequences = tokenizer.texts_to_sequences(train_data)

    y_train_one_hot = to_categorical(train_labels, num_classes=4)

    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    X_train, X_temp, Y_train, Y_temp = train_test_split(padded_sequences, y_train_one_hot, test_size=0.3, random_state=28)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

    lstm_model.fit(X_train, Y_train, X_val, Y_val)
    lstm_model.evaluate(X_test, Y_test)
    lstm_model.make_plots()
    lstm_model.saving_model(model_save_path)


def preprocess_data(df: pd.DataFrame, add_subject: bool, add_domain: bool) -> Tuple[np.ndarray, np.ndarray]:
    combined_data = df['body']

    if add_subject:
        combined_data = df['subject'] + " " + combined_data
    if add_domain:
        df['domain'] = (df['sender'] == df['receiver']).astype(int)

        combined_data = df['domain'].astype(str) + " " + combined_data

    return combined_data.values, df['label'].values




MODELS_PARAMS = [
    # {"model_name": "only-body", "dataset": "../../dataset/processed/final.csv", "add_subject": False, "add_domain": False},
    {"model_name": "stop-words-body", "dataset": "../../dataset/processed/final-with-stop-words.csv", "add_subject": False, "add_domain": False},
    {"model_name": "body-subject-stop", "dataset": "../../dataset/processed/final-with-stop-words.csv", "add_subject": True, "add_domain": False},
    {"model_name": "body-domain-stop", "dataset": "../../dataset/processed/final-with-stop-words-domain-only.csv", "add_subject": False, "add_domain": True},
    {"model_name": "full-data-stop", "dataset": "../../dataset/processed/final-with-stop-words-domain-only.csv", "add_subject": True, "add_domain": True}
]


for params in MODELS_PARAMS:
    data = pd.read_csv(params["dataset"])
    numpy_data, labels = preprocess_data(data, add_subject=params["add_subject"], add_domain=params["add_domain"])

    body_max_length = 1000

    body_model = LSTMModelAdvanced(params['model_name'])
    train_lstm_model(body_model, numpy_data, labels, body_max_length, params["model_name"])



