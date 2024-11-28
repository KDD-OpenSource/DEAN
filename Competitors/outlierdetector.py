import numpy as np
import tensorflow as tf
# from tensorflow import keras
import keras as keras
from keras import backend as K


from sklearn.metrics import roc_auc_score

class OutlierDetector:
    def __init__(self, data, *args, **kwargs):
        self.trained=False
        # self.random_state=random_state

        if not data is None:
            self.train(data)

    def train(self,data):
        """trains the model using the data provided. No hard assumptions about data, but at least most of the datapoints should be normal."""
        raise NotImplementedError

    def predict(self, data):
        """uses the trained model to predict whether the data is normal or not. Returns an array of numeric values, where higher numbers represent a higher likelihood of a sample being abnormal"""
        raise NotImplementedError

    def calculate_auc(self, data, labels):
        """calculates the area under the curve for the given data and labels"""
        self.assert_trained()
        predictions=self.predict(data)
        return roc_auc_score(labels,predictions)


    def is_trained(self):
        return self.trained

    def assert_trained(self):
        if not self.is_trained():
            raise Exception("Model is not trained yet")

    def save_model(self,pth):
        """saves the model to a file"""
        raise NotImplementedError

    def load_model(self,pth):
        """loads the model from a file"""
        raise NotImplementedError

    # def set_seed(self):
    #     if not self.random_state is None:
    #         np.random.seed(self.random_state)
    #         tf.random.set_seed(self.random_state)
    #         keras.utils.set_random_seed(self.random_state)

