from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from utils import text_preprocessing, get_computed_similarities, display_top_n
from doc2VecTransformer import Doc2VecTransformer, doc2Vec_generator
from sklearn.covariance import EllipticEnvelope
import numpy as np

from scipy.stats import multivariate_normal
from scipy.spatial import distance
from scipy.stats import chi2
from os import path
from sklearn.neural_network import MLPRegressor


# We read data and separate good and bad reviews
data = pd.read_csv('amazon_reviews.txt', delimiter="\t")

features = ['RATING', 'REVIEW_TEXT']
data_true = data[  data['LABEL'] == '__label1__' ]
data_shortened = data_true[features]

data_pos = data_shortened[  data_shortened['RATING'] == 5 ]

data_neg = data_shortened[data_shortened['RATING'] == 1]

#_, doc2vec_vectors_bad = doc2Vec_generator(data_neg)

#print( data_pos.head() )
#print( data_neg.head() )

# We generate vectors for the good reviews
#doc2vec_transformer, doc2vec_vectors = doc2Vec_generator(data_pos)

# We learn vector representation for both good and bad reviews
#If you are running for first time delete model and vectors.npy (in case you want to train the model on different data)
doc2vec_transformer, _ = doc2Vec_generator(data_pos)

doc2vec_vectors_positive = doc2vec_transformer.transform(data_pos)
doc2vec_vectors_negative = doc2vec_transformer.transform(data_neg)

auto_encoder = MLPRegressor(hidden_layer_sizes=(
                                                 600,
                                                 150, 
                                                 600,
                                               ))

# We fit autoencoder on the good reviews
auto_encoder.fit(doc2vec_vectors_positive, doc2vec_vectors_positive)

# Perform predictions on both good and bad reviews
predicted_vectors_positive = auto_encoder.predict(doc2vec_vectors_positive)
predicted_vectors_negative = auto_encoder.predict(doc2vec_vectors_negative)

# Get the score of how good our autoencoder fits the good reviews
print(auto_encoder.score(predicted_vectors_positive, doc2vec_vectors_positive))

# Get the score of how good our autoencoder fits the bad reviews
# If bad and good reviews have different representation the score should be lower
print(auto_encoder.score(predicted_vectors_negative, doc2vec_vectors_negative))   







#review_good = "This is a great product!"
#review_bad = "What an absolute rubbish! Such a terrible terrible product! I never hated something like this before!"







#bad_vector = doc2vec_transformer.transform_string(review_bad)

#predicted_bad = auto_encoder.predict(bad_vector)

#print(get_computed_similarities(doc2vec_vectors_bad, predicted_neg_vectors)[:5])



'''
# Testing on single reviews 

review_bad = data_neg.iloc[10, 1]
review_good = data_pos.iloc[5, 1]


reviews = [review_good, review_bad]

review_vector_good = doc2vec_transformer.transform_string(review_good)
review_vector_bad = doc2vec_transformer.transform_string(review_bad) 

test_reviews = np.stack((review_vector_bad.squeeze(), review_vector_good.squeeze()), axis=0)


unique_reviews = get_computed_similarities(doc2vec_vectors_bad, predicted_test_vectors)

for index, similarity in unique_reviews:
    print(f'Review: {reviews[index]} ')
    print(f'Cosine Sim Val: {similarity}')

#display_top_n(data_pos, unique_reviews, 2)'''






