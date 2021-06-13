import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from keras.layers import Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_simple_LSTM_model():
    max_length = 40
    vocab_size = 6000

    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_length))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    # labels = data_df["LABEL"]
    #
    # mySeed = 65  # to receive the same split at each run
    #
    # train_set, test_set, train_labels, test_labels = train_test_split(data_df, labels.values, test_size=0.2,
    #                                                                   random_state=mySeed)  # Partition the dataset to train, validation and test

    train_df = pd.read_csv('Dataset/amazon_reviews_cleaned_train.csv')
    # test_df = pd.read_csv('Dataset/amazon_reviews_cleaned_test.csv')

    train_df = train_df.fillna(' ')
    # test_df = test_df.fillna(' ')

    train_df["all_info"] = train_df["REVIEW_TEXT"]

    tokenizer = Tokenizer(oov_token="<OOV>", num_words=6000)
    tokenizer.fit_on_texts(train_df)

    target = train_df["LABEL"]

    sequences_train = tokenizer.texts_to_sequences(train_df["all_info"])
    # sequences_test = tokenizer.texts_to_sequences(test_df['REVIEW_TEXT'])

    max_length = 40

    padded_train = pad_sequences(sequences_train, padding='post', maxlen=max_length)
    # padded_test = pad_sequences(sequences_test, padding='post', maxlen=max_length)

    X_train, X_test, y_train, y_test = train_test_split(padded_train, target, test_size=0.2)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=15,
                                      verbose=1,
                                      mode="min",
                                      restore_best_weights=True)
    ]

    model = get_simple_LSTM_model()
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(X_train,
                        y_train,
                        epochs=20,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print('Accuracy: ', accuracy_score(y_test, y_pred))
