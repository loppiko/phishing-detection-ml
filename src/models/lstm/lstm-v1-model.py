
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.models.model import Model


class LSTMModelBasic(Model):

    def __init__(self, max_length: int):
        self.model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=max_length),
            LSTM(128, return_sequences=True),  # Pierwsza warstwa LSTM z return_sequences=True
            Dropout(0.5),
            LSTM(128),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # Wyjście dla 4 klas
        ])

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")



def train_lstm_model(lstm_model: LSTMModelBasic, train_data: pd.DataFrame, train_labels: pd.DataFrame, max_length: int) -> None:
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data)

    sequences = tokenizer.texts_to_sequences(train_data)

    y_train_one_hot = to_categorical(train_labels, num_classes=4)

    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    X_train, X_temp, Y_train, Y_temp = train_test_split(padded_sequences, y_train_one_hot, test_size=0.3, random_state=28)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

    lstm_model.fit(X_train, Y_train, X_val, Y_val)
    print(lstm_model.evaluate(X_test, Y_test))


data = pd.read_csv("../../dataset/processed/final.csv")


body = data['body'].values
labels = data['label'].values

body_max_length = 500

# print(body[:5])
# print(labels[:5])

model = LSTMModelBasic(body_max_length)

train_lstm_model(model, body, labels, body_max_length)