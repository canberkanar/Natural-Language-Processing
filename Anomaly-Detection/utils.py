from gensim import utils
import gensim.parsing.preprocessing as gsp


from scipy.spatial.distance import cosine

def key_consine_similarity(tupple):
    return tupple[1]

def get_computed_similarities(vectors, predicted_vectors, reverse=False):
    data_size = len(vectors)
    cosine_similarities = []
    for i in range(data_size):
        cosine_sim_val = cosine(vectors[i], predicted_vectors[i])
        cosine_similarities.append((i, cosine_sim_val))

    return sorted(cosine_similarities, key=key_consine_similarity, reverse=reverse)

def display_top_n(data_df, sorted_cosine_similarities, n=5):
    for i in range(n):
        index, consine_sim_val = sorted_cosine_similarities[i]
        print('Review: ', data_df.iloc[index, 1])  
        print('Cosine Sim Val :', consine_sim_val)
        print('---------------------------------')


filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def text_preprocessing(review):
    review = review.lower()
    review = utils.to_unicode(review)
    for filter in filters:
        review = filter(review)
    
    return review


#review = "Hello World, it is really nice to have you here! @Hey #Hello <p>"

#print(text_preprocessing(review))