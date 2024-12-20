"""
Model evaluation and analysis functions for PPPI.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from .models.tree_models import PPPIRandomForest, PPPIXGBoost, PPPILightGBM, PPPICatBoost
from .models.neural_net import PPPINeuralNet
from .models.ensemble import PPPIVotingEnsemble, PPPIStackingEnsemble

def build_pressure_model(play_features):
    """Build and evaluate multiple pressure prediction models."""
    # Select features for model
    feature_columns = [col for col in play_features.columns 
                      if col not in ['gameId', 'playId', 'causedPressure']]
    
    X = play_features[feature_columns]
    y = play_features['causedPressure']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Balance classes
    smotetomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train_scaled, y_train)
    
    # Initialize models
    models = {
        'Random Forest': PPPIRandomForest(),
        'XGBoost': PPPIXGBoost(),
        'LightGBM': PPPILightGBM(),
        'CatBoost': PPPICatBoost(),
        'Neural Network': PPPINeuralNet(),
        'Voting Ensemble': PPPIVotingEnsemble(),
        'Stacking Ensemble': PPPIStackingEnsemble()
    }
    
    # Train and evaluate models
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_balanced, y_train_balanced, 
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        print(f"Cross-validation ROC AUC scores: {cv_scores}")
        print(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.3f}")
        
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    
    # Analyze feature importance
    importance_analysis = analyze_feature_importance(
        best_model,
        X_train_balanced,
        X_test_scaled,
        y_train_balanced,
        y_test,
        feature_columns
    )
    
    return best_model, scaler, feature_columns, importance_analysis

def analyze_feature_importance(model, X_train, X_test, y_train, y_test, feature_names):
    """Analyze feature importance using multiple methods."""
    importance_results = {}
    
    # Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    importance_results['permutation'] = perm_importance_df
    
    # SHAP Values
    if hasattr(model, 'predict_proba'):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            shap_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(shap_values).mean(0)
            }).sort_values('Importance', ascending=False)
            
            importance_results['shap'] = shap_importance_df
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
    
    # Recursive Feature Elimination
    rfe = RFE(
        estimator=PPPIRandomForest().model,
        n_features_to_select=20
    )
    
    rfe.fit(X_train, y_train)
    rfe_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    importance_results['rfe'] = rfe_importance_df
    
    # Print summary
    print("\nTop 10 Most Important Features (Permutation Importance):")
    print(perm_importance_df.head(10))
    
    if 'shap' in importance_results:
        print("\nTop 10 Most Important Features (SHAP):")
        print(shap_importance_df.head(10))
    
    print("\nSelected Features by RFE:")
    print(rfe_importance_df[rfe_importance_df['Selected']].sort_values('Ranking'))
    
    return importance_results

def calculate_pppi(model, scaler, feature_columns, play_features):
    """Calculate the Pre-snap Pressure Prediction Index."""
    # Select and scale features
    X = play_features[feature_columns]
    X_scaled = scaler.transform(X)
    
    # Calculate PPPI
    pppi = model.predict_proba(X_scaled)[:, 1]
    
    # Add PPPI to features
    play_features_with_pppi = play_features.copy()
    play_features_with_pppi['pppi'] = pppi
    
    # Calculate quartiles
    play_features_with_pppi['pppi_quartile'] = pd.qcut(
        pppi, 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    # Print distribution statistics
    print("\nPPPI Distribution:")
    print(pd.Series(pppi).describe())
    
    # Analyze pressure rate by quartile
    pressure_by_quartile = play_features_with_pppi.groupby('pppi_quartile')['causedPressure'].agg(['mean', 'count'])
    print("\nPressure Rate by PPPI Quartile:")
    print(pressure_by_quartile)
    
    return play_features_with_pppi 