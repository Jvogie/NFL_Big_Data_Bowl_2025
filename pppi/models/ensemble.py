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
        estimators = [
            ('rf', PPPIRandomForest().model),
            ('xgb', PPPIXGBoost().model),
            ('lgb', PPPILightGBM().model),
            ('cat', PPPICatBoost().model)
        ]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1, 1.2, 1.2, 1.2]
        )

class PPPIStackingEnsemble(PPPIBaseModel):
    def __init__(self, cv=5):
        super().__init__()
        base_estimators = [
            ('rf', PPPIRandomForest().model),
            ('xgb', PPPIXGBoost().model),
            ('lgb', PPPILightGBM().model),
            ('cat', PPPICatBoost().model)
        ]
        
        self.model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=PPPILightGBM().model,
            cv=cv,
            n_jobs=-1
        )

def create_weighted_ensemble(models, weights):
    """Create a custom weighted ensemble from multiple models."""
    def weighted_predict_proba(X):
        probas = np.array([model.predict_proba(X) for model in models])
        return np.average(probas, axis=0, weights=weights)
    return weighted_predict_proba 