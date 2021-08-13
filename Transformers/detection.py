import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification


# We want to take the reviews data from the 21k examples, extract the positive data and then use Doc2Vec to get the vectors
data = pd.read_csv('../Anomaly-Detection/amazon_reviews.txt', delimiter="\t")

features = ['RATING', 'REVIEW_TEXT', 'VERIFIED_PURCHASE', 'LABEL']
data_true = data[  data['LABEL'] == '__label1__' ]
data_shortened = data_true[features]

data_shortened['VERIFIED_PURCHASE'] = data_shortened['VERIFIED_PURCHASE'].replace('N', 0)
data_shortened['VERIFIED_PURCHASE'] = data_shortened['VERIFIED_PURCHASE'].replace('Y', 1)

data_shortened['LABEL'] = data_shortened['LABEL'].replace('__label1__', 1)
data_shortened['LABEL'] = data_shortened['LABEL'].replace('__label2__', 0)

data_pos = data_shortened[  data_shortened['RATING'] == 5 ]
data_neg = data_shortened[data_shortened['RATING'] == 1]

print(data_pos.head())

labels = data_pos.pop('LABEL')

print(data_pos.head())

print(data_pos.dtypes)

#tf_dataset= tf.data.Dataset.from_tensor_slices((data_pos.values, labels.values))

#print(tf_dataset.take(5))

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

#x = data_pos.head()['REVIEW_TEXT'].tolist()

#print(x)

tokenized_df = tokenizer(data_pos.head()["REVIEW_TEXT"].tolist(), padding="max_length", truncation=True)

print(type(tokenized_df))
print(tokenized_df.keys())

#y = tokenized_df.data
#print(type(y))
#print(y)
#print(tokenized_df['input_ids'][0][0:5])


train_features = {x: tf.convert_to_tensor(tokenized_df[x]) for x in tokenizer.model_input_names}
print(train_features)

#The dataset now is a tuple of dictionary containint inpud_ids, token_type_ids. attention masks        and  a tf tensor containing the value for the label
train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, labels.head()))

print(train_tf_dataset)
train_tf_dataset = train_tf_dataset.shuffle(5).batch(5)

model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

model.fit(train_tf_dataset, validation_data=train_tf_dataset, epochs=3)

x = next(train_tf_dataset.batch(60_00).as_numpy_iterator())[0]

label = model(x)

print(label)
