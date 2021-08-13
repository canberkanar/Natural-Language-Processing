#  Seminar course Applied Deep Learning for NLP at TU-München

In scope of this research project, we have experimented with state of the art approaches for Natural Language Processing, in order to identify fake product reviews at one of the leader Virtual Merchants of the world. All of our models are trained on the amazon_reviews dataset, which explicity has fake reviews labeled.

### Content

1 - Transfer Learning with Tensorflow Hub Model

- Experiments were conducted on a simple model by using a pre-trained model provided through tensorflow hub. Weight were frozen and unfrozen to observe the different training behaviour.


5 - Transformer Model of distilled Bert

- We enrich the review text with the information of whether a purchase was verified or not and the rating given before feeding it to the transformer network. The network is then fine-tuned to be able to detect fake amazon reviews.

6 - Data Visualization and unsupervised learning

- We use Doc2Vec model to encode review text into 300 dimensional vector. The goal was to find any statistical difference between fake and authentic reviews or good and bad reviews to automatically detect them. However, after reducing the dimesnion to 2D using PCA and t-SNE, we discover that there is no significant statistical difference.
- Inside the folder Anomaly-Detection there are scripts that try to fit normal distribution to the embeddings, as well as an AutoEncoder architecture that tries to learn latent lower level representation of the embeddings.
    
