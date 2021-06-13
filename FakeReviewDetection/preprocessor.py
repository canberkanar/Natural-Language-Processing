# https://github.com/haresrv/Case-Study_Fake-Product-Review-Monitoring-and-Removal

# Expands English contractions in Text (I'd -> I would)
# https://pypi.org/project/contractions/
import contractions

import re  # library to use REGEX Operations to filter unwanted characters
import string

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.model_selection import train_test_split

PRODUCT_CATEGORY_TO_INTEGER = {'Kitchen': 0, 'Home': 1, 'Grocery': 2, 'Sports': 3, 'Jewelry': 4,
                               'Home Entertainment': 5, 'Video DVD': 6, 'Books': 7, 'Shoes': 8, 'PC': 9,
                               'Furniture': 10, 'Video Games': 11, 'Camera': 12, 'Watches': 13, 'Electronics': 14,
                               'Office Products': 15, 'Health & Personal Care': 16, 'Pet Products': 17, 'Baby': 18,
                               'Outdoors': 19, 'Toys': 20, 'Musical Instruments': 21, 'Wireless': 22, 'Luggage': 23,
                               'Apparel': 24, 'Lawn and Garden': 25, 'Automotive': 26, 'Tools': 27, 'Beauty': 28,
                               'Home Improvement': 29}  # To map product


def get_numeric_label(label):  # Turns string labels into integer 0 --> Fake, 1 --> Real
    return int(label[-3])


def text_cleaning(text):  # Data Preprocessing
    text = str(text)
    text = text.lower()  # All text must be converted to lowercase to reduce duplicate detection and processing of the
    # same word
    text = contractions.fix(text)  # Expands all English contractions to reduce duplicate detection and processing of
    # the same expression (who's -> who is)

    # Filtering unwanted characters via Regular Expressions
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"\s+[a-z]\s+", " ", text)
    text = re.sub(r"^[a-z]\s+", " ", text)
    text = re.sub(r"\s+[a-z]$", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text

    doc = nlp(text)

    # tokens = []
    # for token in doc:  # Tokenization
    #     if token.lemma_ != "-PRON-":
    #         temp = token.lemma_.lower().strip()
    #     else:
    #         temp = token.lower_
    #     tokens.append(temp)
    #
    # filtered_tokens = []
    # for token in tokens:
    #     if token not in STOP_WORDS and token not in string.punctuation:
    #         filtered_tokens.append(token)
    # return filtered_tokens


if __name__ == '__main__':
    data_df = pd.read_csv("Dataset/amazon_reviews.txt", index_col=0, delimiter="\t")
    print(data_df.shape)
    print(data_df.LABEL.value_counts())
    print(data_df.head())
    # __label1__ is REAL REVIEW
    # __label2__ is FAKE REVIEW
    # Run on CONDA Terminal to import the model --> python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')

    # Following lines convert string formatted information into integer for better training performance
    # data_df["LABEL"] = data_df["LABEL"].apply(get_numeric_label)  # Turns all labels to Int
    data_df["PRODUCT_CATEGORY"] = data_df["PRODUCT_CATEGORY"].apply(
        lambda x: PRODUCT_CATEGORY_TO_INTEGER[x])  # Turns all categories to int
    data_df["VERIFIED_PURCHASE"] = data_df["VERIFIED_PURCHASE"].apply(
        lambda x: int(x == "Y"))  # Turns all verified purchase cases from char to int

    data_df["REVIEW_TEXT"] = data_df['REVIEW_TITLE'] + " " + data_df["REVIEW_TEXT"]
    data_df.drop("REVIEW_TITLE", inplace=True, axis=1)
    data_df["REVIEW_TEXT"] = data_df['REVIEW_TEXT'].apply(lambda x: text_cleaning(x))

    data_df.drop("PRODUCT_TITLE", axis=1, inplace=True)
    data_df.drop("PRODUCT_ID", axis=1, inplace=True)

    data_df.to_csv("Dataset/amazon_reviews_cleaned_train.csv")

    # X_train, X_test, _, _ = train_test_split(data_df, data_df["LABEL"], test_size=0.2)
    #
    # X_train.to_csv("Dataset/amazon_reviews_cleaned_train.csv")
    # X_test.to_csv("Dataset/amazon_reviews_cleaned_test.csv")
