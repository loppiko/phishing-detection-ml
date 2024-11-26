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

import numpy as np

from src.models.model import Model

# TODO
### 🔷 1. Compare model with and without stopwords
### 2. Use padding to cut additional part of the message (max_len) - Normalizacja długości sekwencji
### 🔷 3. Only domain
###     3a. Only domains
###    🔷 3b. Domain matches
### 🔷 4. LSTM Layers (return_sequences=True)
### 🔷 5. Dropout -> Zapobiega przeuczeniu
### ? Bidirectional
### 6. Hiper-parameters opt.
### 🔷 7. Early Stopping
### 8. Compare all models
### ? Cross validation
### 🔷 ? F1-score -> Not eqauls dataset, Precision/Recall -> Fal-Pos, Pos-Fal have meaning



class LSTMModelAdvanced(Model):
    history = None

    def __init__(self, max_length: int):
        self.model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=max_length),
            LSTM(128),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # Wyjście dla 4 klas
        ])


    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], callbacks=[early_stopping])
        self.history = self.model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)


    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")


    def make_plots(self, x_test: pd.DataFrame, y_test: pd.DataFrame, model_name: str) -> None:
        y_pred_probs = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['legal', 'spam', 'phishing', 'fraud'],
                    yticklabels=['legal', 'spam', 'phishing', 'fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"../../saved-results/lstm/v2/{model_name}-conf-matrix.png")

        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']

        f1 = f1_score(y_true, y_pred, average='weighted')  # średnia ważona F1-score dla wielu klas
        val_f1 = f1_score(y_true, y_pred, average='weighted')

        epochs = range(1, len(accuracy) + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracy, label='Training Accuracy')
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Wykres F1-Score
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [f1] * len(epochs), label='Training F1-Score', linestyle='dashed')
        plt.plot(epochs, [val_f1] * len(epochs), label='Validation F1-Score', linestyle='dashed')
        plt.title('F1-Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.legend()

        plt.savefig(f"../../saved-results/lstm/v2/{model_name}-accuracy.png")



    def saving_model(self, save_path: str) -> None:
        self.model.save(save_path)
        print(f"Model saved to {save_path}")


    @staticmethod
    def loading_model(load_path: str) -> Model:
        return load_model(load_path)


def train_lstm_model(lstm_model: LSTMModelAdvanced, train_data: np.ndarray, train_labels: np.ndarray, max_length: int, model_name: str, model_save_path: str | None = None) -> None:
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data)

    sequences = tokenizer.texts_to_sequences(train_data)

    y_train_one_hot = to_categorical(train_labels, num_classes=4)

    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    X_train, X_temp, Y_train, Y_temp = train_test_split(padded_sequences, y_train_one_hot, test_size=0.3, random_state=28)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

    lstm_model.fit(X_train, Y_train, X_val, Y_val)
    print(lstm_model.evaluate(X_test, Y_test))
    lstm_model.make_plots(X_test, Y_test, model_name)

    if model_save_path is not None:
        lstm_model.saving_model(model_save_path)


def preprocess_data(df: pd.DataFrame, add_subject: bool, add_domain: bool) -> Tuple[np.ndarray, np.ndarray]:
    combined_data = df['body']

    if add_subject:
        combined_data = df['subject'] + " " + combined_data
    if add_domain:
        df['domain'] = (df['sender'] == df['receiver']).astype(int)

        combined_data = df['domain'].astype(str) + " " + combined_data

    return combined_data.values, df['label'].values



data = pd.read_csv("../../dataset/processed/final.csv").sample(n=10000, random_state=29)
# models_names = ["only-body", "stop-words-body", "body-subject", "body-domain", "full-data"]


numpy_data, labels = preprocess_data(data, add_subject=False, add_domain=True)


body_max_length = 1000

body_model = LSTMModelAdvanced(body_max_length)

train_lstm_model(body_model, numpy_data, labels, body_max_length, {model_name}, f"{model_name}.h5")
# train_lstm_model(subject_model, subject, labels, subject_max_length, plot_save_path)
