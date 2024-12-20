"""
Base model classes and common functionality.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class PPPIBaseModel(BaseEstimator, ClassifierMixin):
    """Base class for all PPPI models."""
    
    def __init__(self):
        self.model_ = None
    
    def fit(self, X, y):
        """Fit the model."""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Fit the underlying model
        if hasattr(self, 'model_') and self.model_ is not None:
            self.model_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        # Check is fit had been called
        check_is_fitted(self, ['model_'])
        
        # Input validation
        X = check_array(X)
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        # Check is fit had been called
        check_is_fitted(self, ['model_'])
        
        # Input validation
        X = check_array(X)
        
        return self.model_.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Get parameters from parent class
        params = super().get_params(deep=deep)
        return params
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 