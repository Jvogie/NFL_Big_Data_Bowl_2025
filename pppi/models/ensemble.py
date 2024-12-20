"""
Ensemble model implementations for PPPI.
"""
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from .tree_models import PPPIRandomForest, PPPIXGBoost, PPPILightGBM, PPPICatBoost
from .base import PPPIBaseModel

class PPPIVotingEnsemble(PPPIBaseModel):
    def __init__(self):
        super().__init__()
        self.rf = PPPIRandomForest()
        self.xgb = PPPIXGBoost()
        self.lgb = PPPILightGBM()
        self.cat = PPPICatBoost()
        self.model_ = None
    
    def fit(self, X, y):
        # First fit the base models
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.lgb.fit(X, y)
        self.cat.fit(X, y)
        
        # Create and fit the voting classifier
        self.model_ = VotingClassifier(
            estimators=[
                ('rf', self.rf.model_),
                ('xgb', self.xgb.model_),
                ('lgb', self.lgb.model_),
                ('cat', self.cat.model_)
            ],
            voting='soft',
            weights=[1, 1.2, 1.2, 1.2]
        )
        return super().fit(X, y)

class PPPIStackingEnsemble(PPPIBaseModel):
    def __init__(self, cv=5):
        super().__init__()
        self.cv = cv
        self.rf = PPPIRandomForest()
        self.xgb = PPPIXGBoost()
        self.lgb = PPPILightGBM()
        self.cat = PPPICatBoost()
        self.final_estimator = PPPILightGBM()
        self.model_ = None
    
    def fit(self, X, y):
        # First fit the base models
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.lgb.fit(X, y)
        self.cat.fit(X, y)
        self.final_estimator.fit(X, y)
        
        # Create and fit the stacking classifier with reduced parallelism
        self.model_ = StackingClassifier(
            estimators=[
                ('rf', self.rf.model_),
                ('xgb', self.xgb.model_),
                ('lgb', self.lgb.model_),
                ('cat', self.cat.model_)
            ],
            final_estimator=self.final_estimator.model_,
            cv=self.cv,
            n_jobs=1  # Disable parallelism to avoid CatBoost file issues
        )
        return super().fit(X, y)

def create_weighted_ensemble(models, weights):
    """Create a custom weighted ensemble from multiple models."""
    def weighted_predict_proba(X):
        probas = np.array([model.predict_proba(X) for model in models])
        return np.average(probas, axis=0, weights=weights)
    return weighted_predict_proba 