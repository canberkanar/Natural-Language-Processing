from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_string
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm

import multiprocessing
import numpy as np

from utils import text_preprocessing
from os import path

def doc2Vec_generator(data_df):
    #If model is not saved, create a Doc2Vec model, train it on our data, transform our reviews to vectors then save both model and vectors
    if (not (path.exists("./model")  or path.exists("vectors.npy")  )):
        for row in data_df.itertuples():
            data_df.at[row.Index, 'REVIEW_TEXT'] = text_preprocessing(row.REVIEW_TEXT)

        doc2vec_transformer = Doc2VecTransformer(vector_size=300)

        doc2vec_transformer.fit(data_df)

        doc2vec_vectors = doc2vec_transformer.transform(data_df)

        np.save("./vectors.npy", doc2vec_vectors)
        doc2vec_transformer._model.save("./model")
    #load model and and transform data into vectors
    else:
        #doc2vec_vectors = np.load("./vectors.npy")
        doc2vec_transformer = Doc2VecTransformer(vector_size=300, model=Doc2Vec.load("./model"))
        doc2vec_vectors = doc2vec_transformer.transform(data_df)
    
    return doc2vec_transformer, doc2vec_vectors

class Doc2VecTransformer(BaseEstimator):
    def __init__(self, vector_size=100, model=None, learning_rate=0.02, epochs=20):
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = model
        self.workers = multiprocessing.cpu_count()-1
    
    def fit(self, df_x, df_y=None):
        tagged_x = [ TaggedDocument(  str(row.REVIEW_TEXT).split() , [row.Index]   )     for row in df_x.itertuples()      ]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            training_data = [review for review in tqdm(tagged_x)]
            model.train(skl_utils.shuffle(training_data),  total_examples=len(training_data), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model

        return self
    
    def transform(self, df_x):
        #return np.asmatrix ( np.array( [ self._model.infer_vector(  str(row.REVIEW_TEXT).split()  ) 
        #              for row in df_x.itertuples() ])
        #            )
        return np.array( [ self._model.infer_vector(  str(row.REVIEW_TEXT).split()  ) 
                      for row in df_x.itertuples() ])

    
    def transform_string(self, review):
        review = text_preprocessing(review)
        vector = self._model.infer_vector( str(review).split() )
        return np.array( [vector] )
                    
