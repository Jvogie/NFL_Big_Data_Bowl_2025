"""
Model evaluation and analysis functions for PPPI.
"""

import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import shap
import seaborn as sns

from pppi.models.tree_models import PPPIRandomForest, PPPIXGBoost, PPPILightGBM, PPPICatBoost
from pppi.models.neural_net import PPPINeuralNet
from pppi.models.ensemble import PPPIVotingEnsemble, PPPIStackingEnsemble
from pppi.visualization import (
    plot_roc_curves,
    plot_feature_importance_comparison,
    plot_pppi_distribution,
    plot_play_alignment,
    find_extreme_pppi_plays
)

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
    
    # Scale all data first
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    # Split after scaling to avoid data leakage
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance both training and test sets
    X_train_balanced, y_train_balanced = balance_data(X_train_scaled, y_train)
    X_test_balanced, y_test_balanced = balance_data(X_test_scaled, y_test)
    
    # Convert to DataFrame to preserve feature names
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_scaled.columns)
    X_test_balanced = pd.DataFrame(X_test_balanced, columns=X_test_scaled.columns)
    
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
        
        # Cross-validation on balanced training data
        cv_scores = cross_val_score(
            model, X_train_balanced, y_train_balanced, 
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        print(f"Cross-validation ROC AUC scores: {cv_scores}")
        print(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on full balanced training data
        model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on balanced test set
        y_pred = model.predict(X_test_balanced)
        y_pred_proba = model.predict_proba(X_test_balanced)[:, 1]
        roc_auc = roc_auc_score(y_test_balanced, y_pred_proba)
        
        print("\nClassification Report:")
        print(classification_report(y_test_balanced, y_pred))
        print(f"ROC AUC Score: {roc_auc:.3f}")
        
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Create visualizations using pre-computed results
    model_results = {name: results[name]['model'] for name in results}
    
    # Plot ROC curves using test predictions we already have
    plot_roc_curves({name: results[name]['model'] for name in results}, 
                   X_test_balanced, y_test_balanced)
    plt.savefig('roc_curves.png')
    plt.close()

    # Create comparison table using pre-computed metrics
    comparison_data = []
    for name in results:
        comparison_data.append({
            'Model': name,
            'Train ROC AUC': results[name]['cv_mean'],
            'Train Std Dev': results[name]['cv_std'],
            'Test ROC AUC': results[name]['roc_auc']
        })
    comparison_table = pd.DataFrame(comparison_data).round(3)
    print("\nModel Comparison Table:")
    print(comparison_table)

    # If CatBoost and XGBoost are both present, compare them using pre-computed predictions
    if 'CatBoost' in results and 'XGBoost' in results:
        cat_results = results['CatBoost']
        xgb_results = results['XGBoost']
        top_model_comparison = pd.DataFrame([
            {
                'Model': 'CatBoost',
                'Accuracy': accuracy_score(y_test_balanced, cat_results['predictions']),
                'Precision': precision_score(y_test_balanced, cat_results['predictions']),
                'Recall': recall_score(y_test_balanced, cat_results['predictions']),
                'F1 Score': f1_score(y_test_balanced, cat_results['predictions']),
                'ROC AUC': cat_results['roc_auc']
            },
            {
                'Model': 'XGBoost',
                'Accuracy': accuracy_score(y_test_balanced, xgb_results['predictions']),
                'Precision': precision_score(y_test_balanced, xgb_results['predictions']),
                'Recall': recall_score(y_test_balanced, xgb_results['predictions']),
                'F1 Score': f1_score(y_test_balanced, xgb_results['predictions']),
                'ROC AUC': xgb_results['roc_auc']
            }
        ]).round(3)
        print("\nTop Models Detailed Comparison:")
        print(top_model_comparison)
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    
    # Analyze feature importance
    importance_analysis = analyze_feature_importance(
        best_model,
        X_train_balanced,
        X_test_balanced,
        y_train_balanced,
        y_test_balanced,
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
            plt.savefig('shap_summary.png')
            plt.close()
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
            print("Continuing with other importance metrics...")
    
    # Recursive Feature Elimination
    try:
        base_rf = PPPIRandomForest()
        base_rf.fit(X_train[:100], y_train[:100])  # Fit on a small subset for initialization
        
        rfe = RFE(
            estimator=base_rf.model_,
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
    
    # Create feature importance comparison plot
    plot_feature_importance_comparison(
        importance_results.get('shap', pd.DataFrame()),
        importance_results.get('permutation', pd.DataFrame()),
        importance_results.get('rfe', pd.DataFrame())
    )
    plt.savefig('feature_importance_comparison.png')
    plt.close()
    
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
    
    # Convert pppi to pandas Series for easier manipulation
    pppi_series = pd.Series(pppi, index=X.index)
    
    # Add PPPI to features
    play_features_with_pppi = play_features.copy()
    play_features_with_pppi['pppi'] = pppi_series
    
    # Calculate quartiles
    play_features_with_pppi['pppi_quartile'] = pd.qcut(
        pppi_series, 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    # Print distribution statistics
    print("\nPPPI Distribution:")
    print(pppi_series.describe())
    
    # Analyze pressure rate by quartile with explicit observed parameter
    pressure_by_quartile = play_features_with_pppi.groupby(
        'pppi_quartile', observed=True
    )['causedPressure'].agg(['mean', 'count'])
    
    print("\nPressure Rate by PPPI Quartile:")
    print(pressure_by_quartile)
    
    # Create PPPI distribution plot
    quartiles = {
        1: pppi_series.quantile(0.25),
        2: pppi_series.quantile(0.50),
        3: pppi_series.quantile(0.75)
    }
    
    plot_pppi_distribution(
        pppi_series,
        play_features_with_pppi['causedPressure'],
        quartiles
    )
    plt.savefig('pppi_distribution.png')
    plt.close()
    
    return play_features_with_pppi 

def analyze_extreme_plays(play_features_with_pppi, tracking_data, plays_df):
    """
    Analyze and visualize plays with extreme (high/low) PPPI scores.
    
    Args:
        play_features_with_pppi: DataFrame containing plays with PPPI scores
        tracking_data: DataFrame containing tracking data
        plays_df: DataFrame containing play information
    """
    # Find extreme plays
    low_pppi_plays, high_pppi_plays = find_extreme_pppi_plays(play_features_with_pppi, n_plays=1)
    
    # Get play details
    for play_type, plays in [('Low', low_pppi_plays), ('High', high_pppi_plays)]:
        for _, play in plays.iterrows():
            # Get play info
            play_info = plays_df[
                (plays_df['gameId'] == play['gameId']) & 
                (plays_df['playId'] == play['playId'])
            ].iloc[0]
            
            # Get tracking data for this play
            play_tracking = tracking_data[
                (tracking_data['gameId'] == play['gameId']) & 
                (tracking_data['playId'] == play['playId'])
            ]
            
            # Create visualization
            plot_play_alignment(
                play_tracking,
                play_info,
                pppi_score=play['pppi'],
                title=f"{play_type} PPPI Play Example"
            )
            plt.savefig(f'{play_type.lower()}_pppi_play.png')
            plt.close()
            
            # Print play details
            print(f"\n{play_type} PPPI Play Details:")
            print(f"PPPI Score: {play['pppi']:.3f}")
            print(f"Game ID: {play['gameId']}")
            print(f"Play ID: {play['playId']}")
            print(f"Teams: {play_info['possessionTeam']} vs {play_info['defensiveTeam']}")
            print(f"Play Description: {play_info['playDescription']}")
            print("\nKey Features:")
            # Print top 5 most important features for this play
            feature_cols = [col for col in play.index if col not in ['gameId', 'playId', 'pppi', 'pppi_quartile', 'causedPressure']]
            features_df = pd.DataFrame({
                'Feature': feature_cols,
                'Value': [play[col] for col in feature_cols]
            }).sort_values('Value', ascending=False)
            print(features_df.head().to_string())