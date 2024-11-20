## Dataset:

https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data?select=phishing_email.csv

 

## Project Objective:

The objective of the project is to classify emails using supervised learning methods. Emails will be categorized into several classes such as:

- Spam
- Normal emails
- Phishing
- Fraud

The dataset is composed of several separate sources:

    Enron and Ling Datasets: primarily focused on the core content of emails.
    CEAS, Nazario, Nigerian Fraud, and SpamAssassin Datasets: provide context about the message, such as the sender, recipient, date, etc.

The data will require preprocessing to create a unified database that includes all necessary information. The entire project consists of approximately 85,000 emails.

The preprocessing steps include:

    Assigning labels to emails with similar subjects or content if some labels are missing (Imputation).
    Removing stop words from the email body and converting text to lowercase.

Final columns in the dataset:

    Sender
    Receiver
    Subject 
    Body