from typing import Tuple

import pandas as pd
import json
import logging
import numpy as np

from xgboost import XGBClassifier

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer

from rapidfuzz import process

from src.models.model import Model
from xgboostCallbacks import F1ScoreCallback, HistoryCallback




MAIN_DIR = "/kaggle/working/"

class XGBoostModel(Model):


    def __init__(self, model_name: str, model_category: str) -> None:
        self.model_name = model_name
        self.model_category = model_category
        self.vectorizer = None
        self.known_words = None
        self.cm: list[list[float]] = []
        self.f1_scores: list[float] = []
        self.history = {}
        self.model = XGBClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            verbosity=3
        )


    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        self.known_words = set(x_train.columns)

        f1_score_callback = F1ScoreCallback(x_val, y_val)
        history_callback = HistoryCallback(x_val, y_val)

        self.model.fit(
            x_train,
            y_train,
            verbose=True,
            eval_set=[(x_val, y_val)],
            callbacks=[f1_score_callback, history_callback]
        )

        self.f1_scores = f1_score_callback.get_scores()
        self.history = history_callback.get_history()

    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = self.model.predict(x_test, validate_features=False)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy


    def save(self, model_save_path: str = "") -> None:
        if model_save_path == "":
            model_save_path = f"{MAIN_DIR}/models/xgb/saved/{self.model_category}-{self.model_name}"

        self.model.save_model(f"{model_save_path}.json")
        with open(f"{model_save_path}-known-words.json", "w") as f:
            json.dump(list(self.known_words), f)


    def load_model(self, model_load_path: str) -> None:
        self.model = XGBClassifier()
        self.model.load_model(f"{model_load_path}.json")
        with open(f"{model_load_path}-known-words.json", "r") as f:
            self.known_words = set(json.load(f))


    def preprocess_to_known(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        lemmatizer = WordNetLemmatizer()

        tqdm.pandas()

        def process_text(text):
            words = text.split()
            processed_words = []

            for word in words:
                lemma = lemmatizer.lemmatize(word.lower())
                if lemma in self.known_words:
                    processed_words.append(lemma)
                else:
                    closest_match = process.extractOne(lemma, self.known_words)
                    if closest_match[1] > 80:
                        processed_words.append(closest_match[0])

            return " ".join(processed_words)

        df[column] = df[column].progress_apply(process_text)

        return df


    def convert_to_tfidf(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer()

        logging.info(f"Start preprocess column: {column}")

        tfidf_matrix = self.vectorizer.fit_transform(tqdm(df[column], desc=f'Preprocessing: {column}'))

        logging.info(f"Shape of tfidf_matrix: {tfidf_matrix.shape}")

        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_matrix,
            columns=self.vectorizer.get_feature_names_out(),
            index=df.index
        )

        df = df.drop(columns=[column])
        df = pd.concat([df, tfidf_df], axis=1)

        return df


    def add_missing_columns(self, df: pd.DataFrame, label_name: str) -> pd.DataFrame:
        existing_words = set(df.columns)

        if self.known_words is None:
            self.known_words = set(df.columns)

        missing_words: set[str] = self.known_words - existing_words

        if missing_words:
            missing_columns_df = pd.DataFrame(
                data=0,
                index=df.index,
                columns=list(missing_words)
            )
            df = pd.concat([df, missing_columns_df], axis=1)

        sorted_columns = sorted(self.known_words)

        # Upewnienie się, że kolumna label_name jest na końcu
        if label_name in sorted_columns:
            sorted_columns.remove(label_name)
        sorted_columns.append(label_name)

        # Zastosowanie posortowanej listy kolumn
        df = df[sorted_columns]

        return df


    def create_confusion_matrix(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        y_pred_probs = self.model.predict_proba(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        self.cm = confusion_matrix(y_true, y_pred)


    def make_plots(self) -> None:
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['legal', 'spam', 'phishing', 'fraud'],
                    yticklabels=['legal', 'spam', 'phishing', 'fraud'])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(f"{MAIN_DIR}saved-results/xgb/plots/{self.model_category}/{self.model_name}-conf-matrix.png")
        plt.clf()

        epochs = len(self.history["train_loss"])

        plt.subplot(1, 3, 1)
        plt.plot(range(epochs), self.history['val_loss'], label="Validation Loss", color='blue')
        plt.plot(range(epochs), self.history['train_loss'], label="Training Loss", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over Epochs")

        plt.subplot(1, 3, 3)
        plt.plot(range(epochs), self.history['val_acc'], label="Validation Accuracy", color='blue')
        plt.plot(range(epochs), self.history['train_acc'], label="Training Accuracy", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy over Epochs")

        plt.subplot(1, 3, 2)
        plt.plot(range(len(self.f1_scores)), self.f1_scores, label="Validation F1", color='green')
        plt.xlabel("Epochs")
        plt.ylabel("F1-Score")
        plt.legend()
        plt.title("F1-Score over Epochs")

        plt.tight_layout()
        plt.savefig(f"{MAIN_DIR}saved-results/xgb/plots/{self.model_category}/{self.model_name}-accuracy.png")


logging.basicConfig( level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)


def convert_to_label_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

    return df



def perform_multi_evaluation(xgb_boost_model_list: list[Tuple[XGBoostModel, float]], test_data: pd.DataFrame, test_labels: pd.DataFrame) -> None:
    for xgb_model, model_accuracy in xgb_boost_model_list:
        pass




CHECK_LOAD_DATA = False


MODELS_PARAMS = [
    {"models": [("body-model", "body", 20)], "model_category": "only-body", "dataset": "../../dataset/processed/final.csv", "add_subject": False, "add_domain": False},
    {"models": [("body-model", "body", 20)], "model_category": "only-body-with-stop-words", "dataset": "../../dataset/processed/final-with-stop-words.csv", "add_subject": False, "add_domain": False},
    # {"models": ["body", "subject"], "model_category": "", "dataset": "", "add_subject": True, "add_domain": False},
    # {"models": ["body", "domain"], "model_category": "", "dataset": "", "add_subject": False, "add_domain": True},
    # {"models": ["body", "subject", "domain"], "model_category": "", "dataset": "", "add_subject": True, "add_domain": True}
]


###        ----         Preprocess data

for param in MODELS_PARAMS:

    models: list[Tuple[XGBoostModel, float]] = []

    data = pd.read_csv(param["dataset"])
    label_column = "$label"
    data = data.rename(columns={'label': label_column})

    for model_name, train_column, data_length in param["models"]:
        input_data = data.sample(n=data_length, random_state=43)

        xgb_model = XGBoostModel(model_name, param["model_category"])

        input_data = input_data[[train_column, label_column]]

        tfidf_matrix = xgb_model.convert_to_tfidf(input_data, column=train_column)
        tfidf_matrix = xgb_model.add_missing_columns(tfidf_matrix, label_column)

        train_labels = tfidf_matrix[[label_column]]
        train_data = tfidf_matrix.drop(columns=[label_column])

        X_train, X_temp, Y_train, Y_temp = train_test_split(train_data, train_labels, test_size=0.3, random_state=28)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

        xgb_model.fit(X_train, Y_train, X_val, Y_val)
        xgb_model.create_confusion_matrix(X_test, Y_test)
        xgb_model.make_plots()
        xgb_model.save()

        model_accuracy = xgb_model.evaluate(X_test, Y_test)

        print(model_accuracy)

        models.append((xgb_model, model_accuracy))

    # if X_test and Y_test:
    #     perform_multi_evaluation(models, X_test, Y_test)

###        ----         training model




###        ----         Preprocess testing data

if CHECK_LOAD_DATA:
    xgb_body_model = XGBoostModel("", "")
    xgb_subject_model = XGBoostModel("", "")


    body_tfidf = data[-6000:]
    subject_tfidf = data[-6000:]


    body_tfidf = xgb_body_model.convert_to_tfidf(body_tfidf, column='body')
    subject_tfidf = xgb_subject_model.convert_to_tfidf(subject_tfidf, column='subject')

    body_tfidf = body_tfidf.drop(columns=['sender', 'receiver', 'subject'])
    subject_tfidf = subject_tfidf.drop(columns=['sender', 'receiver', 'subject'])


    body_tfidf = xgb_body_model.add_missing_columns(body_tfidf, '$label')
    subject_tfidf = xgb_subject_model.add_missing_columns(subject_tfidf, '$label')

    body_labels = body_tfidf['$label']
    subject_labels = subject_tfidf['$label']

    body_tfidf = body_tfidf.drop(columns=['$label'])
    subject_tfidf = subject_tfidf.drop(columns=['$label'])
    # subject_tfidf = xgb_subject_model.convert_to_tfidf(X, column='subject')
    # subject_labels = Y


    # TODO
    # Combine 2 models and build the full one
    ###        ----         Testing model

    print(f"Evaluation body: {xgb_body_model.evaluate(body_tfidf, body_labels)}")
    print(f"Evaluation subject: {xgb_subject_model.evaluate(subject_tfidf, subject_labels)}")
    # #

    # body_model = XGBoostModel()
    #
    # body_model.load_model("xgb_body_model")
    #
    # body_tfidf = body_model.preprocess_to_known(data[:1000], column='body')
    # body_tfidf = data[:1000].rename(columns={'label': '$label'})
    # #
    # body_tfidf = body_model.convert_to_tfidf(body_tfidf, column='body')
    # body_tfidf = body_tfidf.drop(columns=['sender', 'receiver', 'subject'])
    # body_tfidf = body_model.add_missing_columns(body_tfidf, '$label')
    # body_labels = body_tfidf['$label']
    # #
    # body_tfidf = body_tfidf.drop(columns=['$label'])
    # #
    # #
    # print(body_model.evaluate(body_tfidf, body_labels))

