"""
Ensemble model implementations for PPPI.
"""
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from .tree_models import PPPIRandomForest, PPPIXGBoost, PPPILightGBM, PPPICatBoost
from .base import PPPIBaseModel

class PPPIVotingEnsemble(PPPIBaseModel):
    def __init__(self):
        super().__init__()
        # Initialize base models
        self.rf = PPPIRandomForest()
        self.xgb = PPPIXGBoost()
        self.lgb = PPPILightGBM()
        self.cat = PPPICatBoost()
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # First fit the base models
        print("Fitting Random Forest...")
        self.rf.fit(X_scaled, y)
        print("Fitting XGBoost...")
        self.xgb.fit(X_scaled, y)
        print("Fitting LightGBM...")
        self.lgb.fit(X_scaled, y)
        print("Fitting CatBoost...")
        self.cat.fit(X_scaled, y)
        
        # Get validation predictions for weight optimization
        rf_pred = self.rf.predict_proba(X_scaled)[:, 1]
        xgb_pred = self.xgb.predict_proba(X_scaled)[:, 1]
        lgb_pred = self.lgb.predict_proba(X_scaled)[:, 1]
        cat_pred = self.cat.predict_proba(X_scaled)[:, 1]
        
        # Calculate correlations between predictions
        preds = np.vstack([rf_pred, xgb_pred, lgb_pred, cat_pred])
        corr_matrix = np.corrcoef(preds)
        
        # Adjust weights based on correlations (less weight for highly correlated models)
        weights = 1 / (corr_matrix.mean(axis=1) + 0.5)  # Add 0.5 to avoid extreme weights
        weights = weights / weights.sum()  # Normalize
        
        # Create and fit the voting classifier with optimized weights
        self.model_ = VotingClassifier(
            estimators=[
                ('rf', self.rf.model_),
                ('xgb', self.xgb.model_),
                ('lgb', self.lgb.model_),
                ('cat', self.cat.model_)
            ],
            voting='soft',
            weights=weights.tolist()
        )
        
        # Fit the ensemble
        print("Fitting Voting Ensemble...")
        return super().fit(X_scaled, y)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)

class PPPIStackingEnsemble(PPPIBaseModel):
    def __init__(self, cv=5):
        super().__init__()
        self.cv = cv
        self.rf = PPPIRandomForest()
        self.xgb = PPPIXGBoost()
        self.lgb = PPPILightGBM()
        self.cat = PPPICatBoost()
        # Use only the parameters that are defined in PPPILightGBM.__init__
        self.final_estimator = PPPILightGBM(
            n_estimators=500,
            learning_rate=0.005,
            num_leaves=16
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.xgb.fit(X_scaled, y)
        self.lgb.fit(X_scaled, y)
        self.cat.fit(X_scaled, y)
        
        estimators = [
            ('rf', self.rf.model_),
            ('xgb', self.xgb.model_),
            ('lgb', self.lgb.model_)
        ]
        
        self.model_ = StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_estimator.model_,
            cv=self.cv,
            n_jobs=-1,
            passthrough=True
        )
        self.model_.fit(X_scaled, y)
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)

def create_weighted_ensemble(models, weights):
    """Create a custom weighted ensemble from multiple models."""
    def weighted_predict_proba(X):
        probas = np.array([model.predict_proba(X) for model in models])
        return np.average(probas, axis=0, weights=weights)
    return weighted_predict_proba 