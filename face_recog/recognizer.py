import numpy as np
import tensorflow as tf
from sklearn.base import TransformerMixin

def standard_normalize_tensor(x):
    mean = x.mean()
    std = x.std()
    img_array = (x - mean)/std    
    return img_array

def minmax_normalize_tensor(x):
    x_max = x.max()
    x_min = x.min()
    x = (x - x_min)/(x_max - x_min)
    return x

class FacenetEmbeddingsTransformer(TransformerMixin):
    def __init__(self, model_path, norm="standard"):
        self.norm = norm
        self.model = tf.keras.models.load_model(model_path)
        
    def normalize(self, X):
        # scale the tensor internally
        if self.norm == 'standard':
            normalizer = standard_normalize_tensor
        elif self.norm == 'minmax':
            normalizer = minmax_normalize_tensor
    
        X_norm = []
        for arr in X:
            X_norm.append(normalizer(arr))
        return np.array(X_norm)


    def fit(self, X):
        return self
    
    def transform(self, X):
        # normalize data
        X = self.normalize(X)

        # generate embeddings
        embeddings = self.model.predict(X)        
        return embeddings
        
