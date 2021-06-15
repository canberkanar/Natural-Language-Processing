from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from utils import text_preprocessing
from doc2VecTransformer import Doc2VecTransformer, doc2Vec_generator
from sklearn.covariance import EllipticEnvelope
import numpy as np

from scipy.stats import multivariate_normal
from scipy.spatial import distance
from scipy.stats import chi2
from os import path


# We want to take the reviews data from the 21k examples, extract the positive data and then use Doc2Vec to get the vectors
data = pd.read_csv('amazon_reviews.txt', delimiter="\t")

features = ['RATING', 'REVIEW_TEXT']
data_true = data[  data['LABEL'] == '__label1__' ]
data_shortened = data_true[features]

data_pos = data_shortened[  data_shortened['RATING'] == 5 ]
data_neg = data_shortened[data_shortened['RATING'] == 1]



#If you are running for first time delete model and vectors.npy (in case you want to train the model on different data)
doc2vec_transformer, doc2vec_vectors = doc2Vec_generator(data_pos)


#Scale vectors to avoid overflow
doc2vec_vectors *= 1000

#fit data to a multivariate gaussian distribution
covariance_matrix = np.cov(doc2vec_vectors,  rowvar=False)
mean_values = np.mean(doc2vec_vectors, axis = 0)

model = multivariate_normal(cov=covariance_matrix,mean=mean_values) #allow_singular=True

#An extreme point to see what happens when you have an outlier
point = [1000 for i in range(300)]
test = np.array(point)


review_good = "This is a great product!"
review_bad = "What an absolute rubbish! Such a terrible terrible product! I never hated something like this before!"

review_vector_good = doc2vec_transformer.transform_string(review_good) * 1000
review_vector_bad = doc2vec_transformer.transform_string(review_bad) * 1000

bad_vectors = doc2vec_transformer.transform(data_neg) * 1000
bad_vector = bad_vectors[0]


#Probability distribution function for several points, we see mean is much more likely than others. However not a huge difference between good and bad review
print(model.pdf(mean_values))
print(model.pdf(review_vector_good) )
print(model.pdf(review_vector_bad))
print(model.pdf(bad_vector))
print(model.pdf(test))


print("")

#Using 0.01% as a cutoff value
threshold_distance =  chi2.ppf( (1-0.01), df=300)

#If any point has a mahalanobis distance greater than threshold should be considered outlier
print(threshold_distance)

print( distance.mahalanobis(review_vector_good, mean_values, np.linalg.inv(covariance_matrix)))
print( distance.mahalanobis(doc2vec_vectors[0], mean_values, np.linalg.inv(covariance_matrix)))
print( distance.mahalanobis(mean_values, mean_values, np.linalg.inv(covariance_matrix)))
print( distance.mahalanobis(review_vector_bad, mean_values, np.linalg.inv(covariance_matrix)))
print( distance.mahalanobis(bad_vector, mean_values, np.linalg.inv(covariance_matrix)))
print( distance.mahalanobis(test, mean_values, np.linalg.inv(covariance_matrix)))
