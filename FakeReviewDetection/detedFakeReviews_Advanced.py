import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.layers.merge import concatenate

import tensorflow_hub as hub

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras import layers, Model

import numpy as np

if __name__ == '__main__':

    #tf.compat.v1.disable_eager_execution()


    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    train_df = pd.read_csv('Dataset/amazon_reviews.txt', index_col=False, delimiter="\t")

    #train_true_df = train_df [  train_df["LABEL"] == 0  ]

    features = ["REVIEW_TEXT", "RATING", "LABEL", "VERIFIED_PURCHASE"]

    train_shortened_df = train_df[features]

    #print(train_shortened_df.head())

    train_shortened_df['VERIFIED_PURCHASE'] = train_shortened_df['VERIFIED_PURCHASE'].replace('N', 0)
    train_shortened_df['VERIFIED_PURCHASE'] = train_shortened_df['VERIFIED_PURCHASE'].replace('Y', 1)


    print(train_shortened_df.head())

    review_array = train_shortened_df["REVIEW_TEXT"].to_numpy(dtype=object)
    labels_array = train_shortened_df["LABEL"].to_numpy(np.int32)
    verified_array = train_shortened_df["VERIFIED_PURCHASE"].to_numpy(np.float32)
    #print(review_array[0:1])
    # We first need to shuffle the data such that both training and validation dataset has both labels
    data_length = len(review_array)
    idx = np.random.permutation(data_length)
    X = review_array[idx]
    verified = verified_array[idx]
    y = labels_array[idx]

    train_ratio = 0.8
    train_data_len = int( train_ratio * data_length )

    # Input divided into train and validation
    x_val = X[train_data_len:]
    verified_val = verified[train_data_len:].reshape((-1, 1))

    partial_x_train = X[:train_data_len]
    #reshape array to be concatenated with the document embedding
    verified_train = verified[:train_data_len].reshape((-1, 1))
    #print(verified_train.shape)



    #output divided into train and validation
    y_val = y[train_data_len:]
    partial_y_train = y[:train_data_len]

    #print(partial_x_train[0:1])

    model_classifier = tf.keras.models.load_model('Models/fakeNewsDetector_Advanced.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

    if not model_classifier:
        # We define how the input layers to our model look like. One input is text and the other if the purchase is verified or not
        #This shape means input is a tf.tensor and its shape is (None,)
        input_text = layers.Input(shape=[], dtype=tf.string)
        #This is the shape of (None, 1)
        input_verified = layers.Input(shape=(1,), dtype=tf.float32)

        print(input_verified.shape)

        model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)


        x = hub_layer(input_text)
        #array = np.ones( (x.shape[0], 1) )
        combined = concatenate([x, input_verified], axis=1)
        # Shape is (None, 51) which is the 50 embedding representation + 1 from verified_purchase!
        print(combined.shape)
        x = Dense(16, activation='relu')(combined)
        x = Dense(1, activation='sigmoid')(x)

        model_classifier = Model(inputs=[input_text, input_verified], outputs=x, name='FakeReviewDetector')

        #Sanity check if our model structure is correct
        model_classifier.summary()

        
        model_classifier.compile(optimizer='adam',
                        loss=tf.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[tf.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


        callbacks = [
            keras.callbacks.EarlyStopping(monitor="accuracy", patience=15,
                                            verbose=1, mode="min", restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(filepath="Models/fakeNewsDetector_Advanced.hdf5", verbose=1, save_best_only=True)
            ]

        history = model_classifier.fit([partial_x_train, verified_train],
                                partial_y_train,
                                epochs=40,
                                batch_size=256,
                                validation_data=([x_val,verified_val], y_val),
                                verbose=1,
                                callbacks=callbacks
                                )

    else:
        results = model_classifier.evaluate([X, verified], y)
        print(f'Accuracy on all data is: {results[1]}')

