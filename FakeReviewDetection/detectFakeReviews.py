import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import tensorflow_hub as hub

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

if __name__ == '__main__':

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    train_df = pd.read_csv('Dataset/amazon_reviews_cleaned_train.csv', index_col=False)

    # X_train, X_test, y_train, y_test = train_test_split(train_df, train_df["LABEL"], test_size=0.2)

    # print("Training entries: {}, test entries: {}".format(len(X_train), len(y_train)))

    # print(X_train["REVIEW_TEXT"][:10])
    # y_train[:10]

    model = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
    hub_layer(train_df["REVIEW_TEXT"][:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)  # Transfer Learning for pre-trained word embeddings
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

    x_val = train_df["REVIEW_TEXT"][:10000]
    partial_x_train = train_df["REVIEW_TEXT"][10000:]

    y_val = train_df["LABEL"][:10000]
    partial_y_train = train_df["LABEL"][10000:]

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="accuracy", patience=15,
                                      verbose=1, mode="min", restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath="Models/fakeNewsDetector.hdf5", verbose=1, save_best_only=True)
    ]

    history = model.fit(partial_x_train.to_numpy(),
                        partial_y_train.to_numpy(),
                        epochs=40,
                        batch_size=256,
                        validation_data=(x_val.to_numpy(), y_val.to_numpy()),
                        verbose=1,
                        callbacks=callbacks)

    results = model.evaluate(train_df["REVIEW_TEXT"].to_numpy(), train_df["LABEL"].to_numpy())

    print(results)
