import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import dtype

import tensorflow_hub as hub

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import numpy as np

if __name__ == '__main__':

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    train_df = pd.read_csv('Dataset/amazon_reviews.txt', index_col=False, delimiter="\t")

    #train_true_df = train_df [  train_df["LABEL"] == 0  ]

    features = ["REVIEW_TEXT", "RATING", "LABEL"]

    train_shortened_df = train_df[features]

    #print(train_shortened_df.head())

    review_array = train_shortened_df["REVIEW_TEXT"].to_numpy(dtype=object)
    labels_array = train_shortened_df["LABEL"].to_numpy(np.int32)
    #print(review_array[0:1])
    # We first need to shuffle the data such that both training and validation dataset has both labels
    data_length = len(review_array)
    idx = np.random.permutation(data_length)
    X = review_array[idx]
    y = labels_array[idx]

    train_ratio = 0.8
    train_data_len = int( train_ratio * data_length )

    x_val = X[train_data_len:]
    partial_x_train = X[:train_data_len]

    y_val = y[train_data_len:]
    partial_y_train = y[:train_data_len]

    #print(y_val)
    #print(len(y_val == 0))
    #print(len(y_val == 1))

    #print(len(x_val == 0))
    #print(len(x_val == 1))



    model_classifier = tf.keras.models.load_model('Models/fakeNewsDetector.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})


    if not model_classifier:
        model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
        #print( hub_layer(review_array[:3]) )


        model_classifier = tf.keras.Sequential()
        model_classifier.add(hub_layer)  # Transfer Learning for pre-trained word embeddings
        #model_classifier.add(Dropout(0.6))
        model_classifier.add(Dense(16, activation='relu'))
        #model_classifier.add(Dropout(0.6))
        model_classifier.add(Dense(1, activation='sigmoid'))

        model_classifier.summary()

        model_classifier.compile(optimizer='adam',
                    loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


        callbacks = [
            keras.callbacks.EarlyStopping(monitor="accuracy", patience=15,
                                        verbose=1, mode="min", restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath="Models/fakeNewsDetector.hdf5", verbose=1, save_best_only=True)
        ]

        history = model_classifier.fit(partial_x_train,
                            partial_y_train,
                            epochs=40,
                            batch_size=256,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            callbacks=callbacks)

        #model_classifier.save("./fakeDetector.hdf5")

    else:
        results = model_classifier.evaluate(review_array,  labels_array)
        #results_valid = model_classifier.evaluate(x_val,  y_val)
        
        print(f'Evaluation on all data has accuracy of: {results[1]} ')
