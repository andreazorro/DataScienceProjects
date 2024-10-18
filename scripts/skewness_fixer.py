from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew
from scipy.special import boxcox1p
import numpy as np
import pandas as pd

class SkewnessFixer(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_threshold=0.5, boxcox_lambda=0.15):
        self.skewness_threshold = skewness_threshold
        self.boxcox_lambda = boxcox_lambda
        self.features_to_transform = []
        self.feature_names = None

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Calculate the skewness of the numerical features
        skewness = X.apply(lambda x: skew(x)).sort_values(ascending=False)
        
        # Filter the numerical features with skewness greater than the threshold
        self.features_to_transform = skewness[abs(skewness) > self.skewness_threshold].index.tolist()
        
        # Store the feature names
        self.feature_names = X.columns.tolist()
        
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X = X.copy()
        # Apply the Box-Cox transformation to the numerical features with skewness greater than the threshold
        for feature in self.features_to_transform:
            X[feature] = boxcox1p(X[feature], self.boxcox_lambda)
        
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)