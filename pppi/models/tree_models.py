"""
Tree-based model implementations for PPPI.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .base import PPPIBaseModel
import numpy as np

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
    def __init__(self, n_estimators=2000, learning_rate=0.005, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model_ = None
    
    def fit(self, X, y):
        neg_count = sum(y == 0)
        pos_count = sum(y == 1)
        scale_pos_weight = neg_count / pos_count
        
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=10,
            gamma=0.2,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.5,
            reg_lambda=2,
            tree_method='hist',
            max_bin=64,
            grow_policy='lossguide',
            random_state=42,
            n_jobs=-1
        )
        return super().fit(X, y)

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
    def __init__(self, iterations=2000, learning_rate=0.005, depth=3):
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