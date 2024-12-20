"""
Tree-based model implementations for PPPI.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .base import PPPIBaseModel

class PPPIRandomForest(PPPIBaseModel):
    def __init__(self, n_estimators=300, max_depth=6, min_samples_leaf=10):
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
            class_weight='balanced_subsample',
            random_state=42
        )
        return super().fit(X, y)

class PPPIXGBoost(PPPIBaseModel):
    def __init__(self, n_estimators=300, learning_rate=0.01, max_depth=4):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model_ = None
    
    def fit(self, X, y):
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=2,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=42
        )
        return super().fit(X, y)

class PPPILightGBM(PPPIBaseModel):
    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=31):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model_ = None
    
    def fit(self, X, y):
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            min_split_gain=1e-3,
            min_child_weight=1e-3,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        return super().fit(X, y)

class PPPICatBoost(PPPIBaseModel):
    def __init__(self, iterations=500, learning_rate=0.02, depth=6):
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
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            class_weights={0: 1, 1: 2},
            random_seed=42,
            verbose=False
        )
        return super().fit(X, y) 