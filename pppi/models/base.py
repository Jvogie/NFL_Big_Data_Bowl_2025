"""
Base model classes and common functionality.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class PPPIBaseModel(BaseEstimator, ClassifierMixin):
    """Base class for all PPPI models."""
    
    def __init__(self):
        self.model = None
    
    def fit(self, X, y):
        """Fit the model."""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"model": self.model}
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 