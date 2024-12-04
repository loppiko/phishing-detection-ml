from xgboost import callback

import pandas as pd
import json

import xgboost
from xgboost import XGBClassifier

from tqdm import tqdm
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from rapidfuzz import process

from src.models.model import Model


### TODO

### Starter: XGBoost ✅

### Transformers: BERT

### Models: LSTM



class XGBoostModel(Model):

    def __init__(self, number_of_estimators: int = 1000, max_depth: int = 10, learning_rate: float = 0.1) -> None:
        self.vectorizer = None
        self.known_words = None
        self.model = XGBClassifier(
            n_estimators=number_of_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            verbosity=3
        )


    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        self.known_words = set(x_train.columns)

        self.model.fit(
            x_train,
            y_train,
            verbose=True,
            eval_set=[(x_val, y_val)]
        )


    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = self.model.predict(x_test, validate_features=False)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy


    def save(self, model_save_path: str) -> None:
        self.model.save_model(f"{model_save_path}.json")
        with open(f"{model_save_path}-known-words.json", "w") as f:
            json.dump(list(self.known_words), f)


    def load_model(self, model_load_path: str) -> None:
        self.model = XGBClassifier()
        self.model.load_model(f"{model_load_path}.json")
        with open(f"{model_load_path}-known-words.json", "r") as f:
            self.known_words = set(json.load(f))


    def plot_model(self, save_path: str) -> None:
        xgboost.plot_importance(self.model)
        plt.savefig(save_path)


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

    def create_confusion_matrix(self):
        pass

    def make_plots(self) -> None:
        pass


logging.basicConfig( level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)


def convert_to_label_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

    return df


def train_xgb_model(xgb_model: XGBoostModel, train_data: pd.DataFrame, train_labels: pd.DataFrame, model_save_path: str) -> None:
    X_train, X_temp, Y_train, Y_temp = train_test_split(train_data, train_labels, test_size=0.3, random_state=28)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=28)

    # train_columns = set(X_train.shape)
    # predict_columns = set(X_test.shape)

    xgb_model.fit(X_train, Y_train, X_val, Y_val)
    xgb_model.save(model_save_path)
    print(xgb_model.evaluate(X_test, Y_test))




data = pd.read_csv("src/dataset/processed/final.csv").sample(frac=1, random_state=43)
data = data.reset_index(drop=True)[:16000]
data = data.rename(columns={'label': '$label'})

# X = data.drop(columns=['label'])
# Y = data['label']




xgb_body_model = XGBoostModel()
xgb_subject_model = XGBoostModel()

###        ----         Preprocess data

body_tfidf = data[:2000]
subject_tfidf = data[:8000]

body_tfidf = xgb_body_model.convert_to_tfidf(body_tfidf, column='body')
subject_tfidf = xgb_subject_model.convert_to_tfidf(subject_tfidf, column='subject')

subject_tfidf = subject_tfidf.drop(columns=['sender', 'receiver', 'body'])
body_tfidf = body_tfidf.drop(columns=['sender', 'receiver', 'subject'])


print('$label' in body_tfidf.columns, '$label' in subject_tfidf.columns)
body_tfidf = xgb_body_model.add_missing_columns(body_tfidf, '$label')
body_labels = body_tfidf['$label']
body_tfidf = body_tfidf.drop(columns=['$label'])

subject_tfidf = xgb_subject_model.add_missing_columns(subject_tfidf, '$label')
subject_labels = subject_tfidf['$label']
subject_tfidf = subject_tfidf.drop(columns=['$label'])

###        ----         training model

train_xgb_model(xgb_body_model, body_tfidf, body_labels, model_save_path="xgb_body_model")
train_xgb_model(xgb_subject_model, subject_tfidf, subject_labels, model_save_path="xgb_subject_model")


###        ----         Preprocess testing data

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

