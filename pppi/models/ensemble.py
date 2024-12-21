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
        # Initialize base models
        self.rf = PPPIRandomForest()
        self.xgb = PPPIXGBoost()
        self.lgb = PPPILightGBM()
        self.cat = PPPICatBoost()
        if hasattr(self.cat.model_, 'set_params'):
            self.cat.model_.set_params(logging_level='Silent', 
                                     allow_writing_files=False)
        self.final_estimator = PPPILightGBM(
            n_estimators=500,
            learning_rate=0.005,
            num_leaves=16
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # First fit base models individually
            print("Fitting base models...")
            self.rf.fit(X_scaled, y)
            self.xgb.fit(X_scaled, y)
            self.lgb.fit(X_scaled, y)
            
            # Initialize list of estimators
            estimators = [
                ('rf', self.rf.model_),
                ('xgb', self.xgb.model_),
                ('lgb', self.lgb.model_)
            ]
            
            # Try to fit CatBoost
            try:
                print("Fitting CatBoost...")
                self.cat.fit(X_scaled, y)
                if self.cat.model_ is not None:
                    estimators.append(('cat', self.cat.model_))
            except Exception as e:
                print(f"Warning: CatBoost fitting failed, continuing without it: {str(e)}")
            
            # Fit final estimator with conservative parameters
            print("Fitting final estimator...")
            self.final_estimator.fit(X_scaled, y)
            
            if self.final_estimator.model_ is None:
                raise ValueError("Final estimator failed to fit")
            
            # Create and fit the stacking classifier
            print("Creating stacking ensemble...")
            self.model_ = StackingClassifier(
                estimators=estimators,
                final_estimator=self.final_estimator.model_,
                cv=self.cv,
                n_jobs=1,
                verbose=0,
                passthrough=True  # Added to include original features
            )
            
            # Fit the stacking ensemble
            print("Fitting stacking ensemble...")
            return super().fit(X_scaled, y)
            
        except Exception as e:
            print(f"Warning: Stacking ensemble failed: {str(e)}")
            print("Falling back to voting ensemble...")
            
            # Create a simpler voting ensemble as fallback
            estimators = []
            if self.rf.model_ is not None:
                estimators.append(('rf', self.rf.model_))
            if self.xgb.model_ is not None:
                estimators.append(('xgb', self.xgb.model_))
            if self.lgb.model_ is not None:
                estimators.append(('lgb', self.lgb.model_))
            
            if not estimators:
                raise ValueError("No base models were successfully fitted")
            
            self.model_ = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            return super().fit(X_scaled, y)
    
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