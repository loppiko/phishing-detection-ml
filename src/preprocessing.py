import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer


# TODO - Final columns in dataset: Sender, Receiver, Subject, Body

# TODO
# 0 - Normal emails ()
# 1 - Spam
# 2 - Phishing
# 3 - Fraud

# TODO
# Wypróbuj stemming na email body
# Wypróbuj samych domen w nazwach


def prepare_nltk() -> None:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def preprocess_dataset_subject(subject: str) -> str | None:
    
    if not isinstance(subject, str) or subject == '':
        return None
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    subject = subject.lower()
    subject = re.sub(r'[^a-zA-Z\s]', '', subject)

    # Stop_words
    subject = " ".join(word for word in subject.split() if word not in stop_words_set)
    subject = " ".join([lemmatizer.lemmatize(word) for word in subject.split()])  # 4. Lematyzacja

    if subject == '':
        return None

    return subject


def preprocess_dataset_text(text: str) -> str | None:

    if not isinstance(text, str):
        return None
    
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    ### Remove garbage
    text = re.sub(r'\b([a-zA-Z])\1{4,}\b', '', text)  # Same letter more than 5 times
    text = re.sub(r'\b\w{15,}\b', '', text)  # Words longer than 15 chars

    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words_set)

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    if text == '':
        return None

    return text


def preprocess_dataset_sender_receiver(data: str) -> str | None:
    
    if not isinstance(data, str):
        return None

    data = data.strip().lower()
    data = data.split('@')[-1]        # TODO: Only domain

    match = re.match(r'^[a-z0-9._-]*', data)  # Dozwolone: litery, cyfry, ., _, -

    if match:
        data = match.group(0)

    if data == '':
            return None

    return data

def manage_dataset_preprocess(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    if 'sender' not in df.columns:
        df['sender'] = None
    if 'receiver' not in df.columns:
        df['receiver'] = None
    
    df = df[['subject', 'body', 'sender', 'receiver',  'label']]
    df['body'] = df['body'].apply(preprocess_dataset_text)
    df['subject'] = df['subject'].apply(preprocess_dataset_subject)
    df['receiver'] = df['receiver'].apply(preprocess_dataset_sender_receiver)
    df['sender'] = df['sender'].apply(preprocess_dataset_sender_receiver)

    return df


def impute_missing_values(df: pd.DataFrame, column: str) -> None:
    mode_value = df[column].mode()[0]
    df.fillna({column: mode_value}, inplace=True)



prepare_nltk()
stop_words_set = set(nltk.corpus.stopwords.words('english'))


list_of_spam_dataset = [
    "dataset/non-processed/SpamAssasin.csv",
    "dataset/non-processed/Enron.csv",
    "dataset/non-processed/Ling.csv"
]
list_of_fraud_dataset = [
    "dataset/non-processed/Nigerian_Fraud.csv"
]
list_of_phishing_dataset = [
    "dataset/non-processed/CEAS_08.csv",
    "dataset/non-processed/Nazario.csv"
]


final_data_frame = pd.DataFrame()

for dataset_path in list_of_spam_dataset:

    df = manage_dataset_preprocess(dataset_path)
    df['label'] = df['label'].map({0: 0, 1: 1})
    final_data_frame = pd.concat([final_data_frame, df], ignore_index=True)

for dataset_path in list_of_phishing_dataset:
    
    df = manage_dataset_preprocess(dataset_path)
    df['label'] = df['label'].map({0: 0, 1: 2})
    final_data_frame = pd.concat([final_data_frame, df], ignore_index=True)


for dataset_path in list_of_fraud_dataset:
    
    df = manage_dataset_preprocess(dataset_path)
    df['label'] = df['label'].map({0: 0, 1: 3})
    final_data_frame = pd.concat([final_data_frame, df], ignore_index=True)


# print(f"subject: {final_data_frame['subject'].isna().sum()}")
# print(f"body: {final_data_frame['body'].isna().sum()}")
# print(f"sender: {final_data_frame['sender'].isna().sum()}")
# print(f"receiver: {final_data_frame['receiver'].isna().sum()}")
# print(f"label: {final_data_frame['label'].isna().sum()}")

impute_missing_values(final_data_frame, "subject")
impute_missing_values(final_data_frame, "body")
impute_missing_values(final_data_frame, "sender")
impute_missing_values(final_data_frame, "receiver")


print(f"subject: {final_data_frame['subject'].isna().sum()}")
print(f"body: {final_data_frame['body'].isna().sum()}")
print(f"sender: {final_data_frame['sender'].isna().sum()}")
print(f"receiver: {final_data_frame['receiver'].isna().sum()}")

print(final_data_frame['sender'].head(5))
print(final_data_frame['receiver'].head(5))

final_data_frame.to_csv("dataset/processed/final-domain-only.csv", index=False)

# Filtrujemy dane i wybieramy 5 pierwszych wierszy, gdzie kolumna 'label' jest równa 1
# filtered_rows = df[df['label'] == 1]



# Wyświetlamy 5 pierwszych wierszy
# print(filtered_rows.head(5))

