"""
Tree-based model implementations for PPPI.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .base import PPPIBaseModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score

class PPPIRandomForest(PPPIBaseModel):
    def __init__(self, n_estimators=2000, max_depth=3, min_samples_leaf=100):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model_ = None
    
    def fit(self, X, y):
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=200,
            max_features=0.5,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        return super().fit(X, y)

class PPPIXGBoost(PPPIBaseModel):
    def __init__(self):
        super().__init__()
        self.model_ = None
        self.classes_ = None
    
    def fit(self, X, y, **fit_params):
        """Fit the XGBoost model with optimized hyperparameters."""
        # Store classes
        self.classes_ = np.unique(y)
        
        # Calculate class weights
        neg_count = sum(y == 0)
        pos_count = sum(y == 1)
        scale_pos_weight = neg_count / pos_count
        
         #Use the best parameters found by Optuna (gives best test roc auc score of .844)
        params = {
            'n_estimators': 11042,
            'max_depth': 7,
            'learning_rate': 0.0040136707521162595,
            'min_child_weight': 1,
            'subsample': 0.6782442958129034,
            'colsample_bytree': 0.7386370052106817,
            'colsample_bylevel': 0.7295932837471293,
            'colsample_bynode': 0.7763947620936577,
            'gamma': 0.25357863483959975,
            'reg_alpha': 0.7504589884121906,
            'reg_lambda': 1.8796183011023513,
            'max_leaves': 115,
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'random_state': 42,
            'n_jobs': -1,
            'enable_categorical': True,
            'scale_pos_weight': scale_pos_weight
        }
        
        # Train model with optimized parameters
        self.model_ = XGBClassifier(**params)
        self.model_.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        check_is_fitted(self, ['model_', 'classes_'])
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels for X."""
        check_is_fitted(self, ['model_', 'classes_'])
        return self.model_.predict(X)

class PPPILightGBM(PPPIBaseModel):
    def __init__(self, n_estimators=2000, learning_rate=0.005, num_leaves=16):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model_ = None
    
    def fit(self, X, y):
        neg_count = sum(y == 0)
        pos_count = sum(y == 1)
        class_weight = {0: 1.0, 1: neg_count/pos_count}
        
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=3,
            min_child_samples=50,
            subsample=0.6,
            colsample_bytree=0.6,
            min_split_gain=0.1,
            min_child_weight=5,
            path_smooth=1,
            extra_trees=True,
            class_weight=class_weight,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        return super().fit(X, y)

class PPPICatBoost(PPPIBaseModel):
    def __init__(self, iterations=5000, learning_rate=0.005, depth=3):
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.model_ = None
    
    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=5,
            bootstrap_type='Bernoulli',
            subsample=0.6,
            rsm=0.6,
            min_data_in_leaf=50,
            leaf_estimation_iterations=10,
            random_strength=1,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            boost_from_average=True,
            langevin=True
        )
        return super().fit(X, y) 