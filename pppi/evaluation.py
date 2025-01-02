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
from imblearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from pppi.models.tree_models import PPPIRandomForest, PPPIXGBoost, PPPILightGBM, PPPICatBoost
from pppi.models.neural_net import PPPINeuralNet
from pppi.models.ensemble import PPPIVotingEnsemble, PPPIStackingEnsemble

def balance_data(X, y):
    pipeline = Pipeline([
        ('smote_tomek', SMOTETomek(random_state=42))
    ])
    X_balanced, y_balanced = pipeline.fit_resample(X, y)
    return X_balanced, y_balanced

def build_pressure_model(play_features, selected_models=None):
    """Build and evaluate multiple pressure prediction models.
    
    Args:
        play_features: DataFrame containing the features
        selected_models: List of model names to train. If None, trains all models.
                        Valid options: ['random_forest', 'xgboost', 'lightgbm', 
                        'catboost', 'neural_net', 'voting', 'stacking']
    """
    # Select features for model
    feature_columns = [col for col in play_features.columns 
                      if col not in ['gameId', 'playId', 'causedPressure']]
    
    X = play_features[feature_columns].copy()
    y = play_features['causedPressure']
    
    # Handle NaN values before splitting
    for col in X.columns:
        if X[col].isna().any():
            if np.issubdtype(X[col].dtype, np.number):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
    
    # Print NaN statistics
    nan_cols = X.isna().sum()
    if nan_cols.any():
        print("\nNaN values found in features:")
        print(nan_cols[nan_cols > 0])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features while preserving feature names
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Balance classes
    X_balanced, y_balanced = balance_data(X_train_scaled, y_train)
    
    # Convert to DataFrame to preserve feature names
    X_balanced = pd.DataFrame(X_balanced, columns=X_train_scaled.columns)
    
    # Define all available models
    all_models = {
        'random_forest': ('Random Forest', PPPIRandomForest()),
        'xgboost': ('XGBoost', PPPIXGBoost()),
        'lightgbm': ('LightGBM', PPPILightGBM()),
        'catboost': ('CatBoost', PPPICatBoost()),
        'neural_net': ('Neural Network', PPPINeuralNet()),
        'voting': ('Voting Ensemble', PPPIVotingEnsemble()),
        'stacking': ('Stacking Ensemble', PPPIStackingEnsemble())
    }
    
    # Select models to train
    if selected_models is None:
        # Use all models except ensembles by default
        selected_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_net']
    
    # Validate selected models
    invalid_models = set(selected_models) - set(all_models.keys())
    if invalid_models:
        raise ValueError(f"Invalid model selections: {invalid_models}. "
                        f"Valid options are: {list(all_models.keys())}")
    
    # Initialize selected models
    models = {all_models[name][0]: all_models[name][1] for name in selected_models}
    
    # Train and evaluate models
    results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Cross-validation
        # if needed this was working: cv=cv, scoring='roc_auc', n_jobs=-1
        cv_scores = cross_val_score(
            model, X_balanced, y_balanced, 
            cv=cv, 
            scoring='roc_auc', 
            n_jobs=-1,
            fit_params={'verbose': False}
        )
        print(f"Cross-validation ROC AUC scores: {cv_scores}")
        print(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        model.fit(X_balanced, y_balanced)
        
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
        X_balanced,
        X_test_scaled,
        y_balanced,
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
            # For ensemble models, use the first base model for SHAP analysis
            if isinstance(model, (PPPIVotingEnsemble, PPPIStackingEnsemble)):
                base_model = model.rf.model_  # Use Random Forest as base for SHAP
            else:
                base_model = model.model_
            
            explainer = shap.TreeExplainer(base_model)
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
            print("Continuing with other importance metrics...")
    
    # Recursive Feature Elimination
    try:
        base_rf = PPPIRandomForest()
        base_rf.fit(X_train[:100], y_train[:100])  # Fit on a small subset for initialization
        
        rfe = RFE(
            estimator=base_rf.model_,  # Use model_ instead of model
            n_features_to_select=20
        )
        
        rfe.fit(X_train, y_train)
        rfe_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        importance_results['rfe'] = rfe_importance_df
        
    except Exception as e:
        print(f"Warning: RFE analysis failed: {e}")
        print("Continuing with other importance metrics...")
    
    # Print summary
    print("\nTop 10 Most Important Features (Permutation Importance):")
    print(perm_importance_df.head(10))
    
    if 'shap' in importance_results:
        print("\nTop 10 Most Important Features (SHAP):")
        print(importance_results['shap'].head(10))
    
    if 'rfe' in importance_results:
        print("\nSelected Features by RFE:")
        print(rfe_importance_df[rfe_importance_df['Selected']].sort_values('Ranking'))
    
    return importance_results

def calculate_pppi(model, scaler, feature_columns, play_features):
    """Calculate the Pre-snap Pressure Prediction Index."""
    # Select and scale features
    X = play_features[feature_columns].copy()
    
    # Handle any NaN values before scaling
    for col in X.columns:
        if X[col].isna().any():
            # Fill NaN with median for numeric columns
            if np.issubdtype(X[col].dtype, np.number):
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            else:
                # For non-numeric columns, fill with mode
                mode_val = X[col].mode()
                X[col] = X[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
    
    # Print NaN statistics before filling
    nan_cols = play_features[feature_columns].isna().sum()
    nan_cols = nan_cols[nan_cols > 0]  # Only show columns with NaN values
    if not nan_cols.empty:
        print("\nNaN values found in features before processing:")
        print(nan_cols)
        print("\nNaN handling strategy:")
        for col in nan_cols.index:
            if np.issubdtype(X[col].dtype, np.number):
                print(f"- {col}: Filled with {'0' if pd.isna(X[col].median()) else 'median'}")
            else:
                print(f"- {col}: Filled with mode or 'Unknown'")
    
    # Scale features while preserving feature names
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
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
    
    # Analyze pressure rate by quartile with explicit observed parameter
    pressure_by_quartile = play_features_with_pppi.groupby(
        'pppi_quartile', observed=True
    )['causedPressure'].agg(['mean', 'count'])
    
    print("\nPressure Rate by PPPI Quartile:")
    print(pressure_by_quartile)
    
    return play_features_with_pppi 