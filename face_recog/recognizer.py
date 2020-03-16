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
        self.model_path = model_path
        self.load_model(self.model_path)
        
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        else:
            self.model_path = model_path

        self.model = tf.keras.models.load_model(model_path)

    def discard_model(self):
        self.model = None

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


    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # normalize data
        X = self.normalize(X)

        # generate embeddings
        embeddings = self.model.predict(X)        
        return embeddings
        
