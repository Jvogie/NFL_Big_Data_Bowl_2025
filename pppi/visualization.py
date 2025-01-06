"""
Visualization functions for PPPI analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score

def plot_roc_curves(models_dict, X, y):
    """
    Plot ROC curves for all models with confidence intervals.
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        X: Feature matrix
        y: True labels
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for (name, model), color in zip(models_dict.items(), colors):
        # Calculate ROC curve
        y_pred = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve with thicker line for CatBoost but no confidence interval
        if name == 'CatBoost':
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')
        else:
            plt.plot(fpr, tpr, color=color, lw=1,
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    return plt

def create_model_comparison_table(models_dict, X_train, X_test, y_train, y_test):
    """
    Create a DataFrame comparing model performances.
    
    Args:
        models_dict: Dictionary of {model_name: model_object}
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        pandas DataFrame with model comparison metrics
    """
    results = []
    
    for name, model in models_dict.items():
        # Get training scores from cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
        
        # Get test score
        y_test_pred = model.predict_proba(X_test)[:, 1]
        test_score = roc_auc_score(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train ROC AUC': cv_scores.mean(),
            'Train Std Dev': cv_scores.std(),
            'Test ROC AUC': test_score
        })
    
    return pd.DataFrame(results).round(3)

def compare_top_models(catboost_model, xgboost_model, X_test, y_test):
    """
    Create comparison table for CatBoost and XGBoost.
    
    Args:
        catboost_model: Trained CatBoost model
        xgboost_model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        pandas DataFrame with detailed model comparison
    """
    models = {'CatBoost': catboost_model, 'XGBoost': xgboost_model}
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        })
    
    return pd.DataFrame(results).round(3)

def plot_feature_importance_comparison(shap_values, perm_importance, rfe_results):
    """
    Create a comparison plot of different feature importance methods.
    
    Args:
        shap_values: DataFrame with SHAP importance values
        perm_importance: DataFrame with permutation importance values
        rfe_results: DataFrame with RFE results
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Get top 15 features from each method
    n_features = 15
    shap_features = set(shap_values['Feature'].head(n_features)) if not shap_values.empty else set()
    perm_features = set(perm_importance['Feature'].head(n_features)) if not perm_importance.empty else set()
    rfe_features = set(rfe_results[rfe_results['Selected']]['Feature'].head(n_features)) if not rfe_results.empty else set()
    
    # Combine all unique features
    features = list(shap_features | perm_features | rfe_features)
    
    if not features:
        print("No features found for importance comparison")
        return plt
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(index=features)
    
    # Add values for each method
    if not shap_values.empty:
        shap_dict = dict(zip(shap_values['Feature'], shap_values['Importance']))
        comparison['SHAP'] = comparison.index.map(lambda x: shap_dict.get(x, 0))
    
    if not perm_importance.empty:
        perm_dict = dict(zip(perm_importance['Feature'], perm_importance['Importance']))
        comparison['Permutation'] = comparison.index.map(lambda x: perm_dict.get(x, 0))
    
    if not rfe_results.empty:
        # Create linear scale for RFE rankings
        rfe_sorted = rfe_results.sort_values('Ranking').head(n_features)
        rfe_dict = {}
        for i, (_, row) in enumerate(rfe_sorted.iterrows()):
            rfe_dict[row['Feature']] = 1 - (i / n_features)
        comparison['RFE'] = comparison.index.map(lambda x: rfe_dict.get(x, 0))
    
    # Normalize SHAP and Permutation columns to [0, 1] range
    for col in ['SHAP', 'Permutation']:
        if col in comparison.columns and comparison[col].sum() > 0:
            comparison[col] = comparison[col] / comparison[col].max()
    
    # Sort by average importance
    comparison['Mean'] = comparison.mean(axis=1)
    comparison = comparison.sort_values('Mean', ascending=True).drop('Mean', axis=1)
    
    # Plot
    ax = comparison.plot(kind='barh', figsize=(12, max(8, len(features) * 0.3)))
    plt.title('Feature Importance Comparison (Normalized)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    return plt

def plot_pppi_distribution(pppi_values, pressure_rates, quartiles):
    """
    Create density plot of PPPI distribution with quartiles and pressure rates.
    
    Args:
        pppi_values: Series of PPPI scores
        pressure_rates: Series of actual pressure outcomes
        quartiles: Dictionary of quartile boundaries
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot density
    sns.kdeplot(data=pppi_values, fill=True, alpha=0.5)
    
    # Add quartile lines
    for q, v in quartiles.items():
        plt.axvline(x=v, color='red', linestyle='--', alpha=0.5)
        plt.text(v, plt.ylim()[1], f'Q{q}', rotation=0)
    
    # Add pressure rate overlay
    ax2 = plt.twinx()
    sns.regplot(x=pppi_values, y=pressure_rates, scatter=False, color='green', ax=ax2)
    
    plt.title('PPPI Distribution with Pressure Rates')
    plt.xlabel('PPPI Score')
    plt.ylabel('Density / Pressure Rate')
    plt.grid(True, alpha=0.3)
    return plt

def plot_play_alignment(tracking_data, play_info, pppi_score=None, title=None):
    """
    Create a football field visualization of pre-snap alignment.
    """
    plt.figure(figsize=(15, 10))
    
    # Create football field
    field = plt.Rectangle((0, 0), 100, 53.3, color='darkgreen', alpha=0.3)
    plt.gca().add_patch(field)
    
    # Add yard lines and numbers
    for yard in range(0, 101, 5):
        alpha = 0.4 if yard % 10 == 0 else 0.2
        plt.axvline(yard, color='white', alpha=alpha, linestyle='-' if yard % 10 == 0 else '--')
        if yard % 10 == 0 and yard > 0 and yard < 100:
            # Calculate yard number (going up to 50 and back down)
            yard_number = min(yard, 100 - yard)
            plt.text(yard, 2, str(yard_number), color='white', alpha=0.6, ha='center', fontsize=8)
            plt.text(yard, 51.3, str(yard_number), color='white', alpha=0.6, ha='center', fontsize=8)
    
    # Add line of scrimmage and first down line (with lower zorder)
    los = play_info['absoluteYardlineNumber']
    plt.axvline(los, color='yellow', linestyle='-', alpha=0.5, linewidth=2, label='Line of Scrimmage', zorder=1)
    
    # Calculate first down line (subtract yards to go since we're going right to left)
    first_down = los - play_info['yardsToGo']
    first_down = max(first_down, 0)  # Don't go below 0
    plt.axvline(first_down, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='First Down Line', zorder=1)
    
    print("\nDebug - Field markings:")
    print(f"Line of scrimmage: {los}")
    print(f"Yards to go: {play_info['yardsToGo']}")
    print(f"First down line: {first_down}")
    
    # Get all frames before snap
    pre_snap_frames = tracking_data[tracking_data['frameType'] == 'BEFORE_SNAP']
    if len(pre_snap_frames) == 0:
        print("No pre-snap frames found!")
        return plt
    
    # Get the last frame ID
    last_frame_id = pre_snap_frames['frameId'].max()
    
    # Get all rows from the last frame
    last_frame = pre_snap_frames[pre_snap_frames['frameId'] == last_frame_id].copy()
    
    # Filter out the football
    player_frame = last_frame[~last_frame['displayName'].str.contains('football', case=False, na=False)]
    
    # Separate offensive and defensive players
    offense = player_frame[player_frame['club'] == play_info['possessionTeam']]
    defense = player_frame[player_frame['club'] == play_info['defensiveTeam']]
    ball = last_frame[last_frame['displayName'].str.contains('football', case=False, na=False)]
    
    # Plot players with different markers and colors
    if not offense.empty:
        plt.scatter(offense['x'], offense['y'], color='blue', s=150, label='Offense', marker='o', zorder=3)
    
    if not defense.empty:
        plt.scatter(defense['x'], defense['y'], color='red', s=150, label='Defense', marker='^', zorder=3)
    
    if not ball.empty:
        plt.scatter(ball['x'], ball['y'], color='brown', s=100, label='Ball', marker='s', zorder=4)
    
    # Set plot limits and labels
    plt.xlim(-5, 105)
    plt.ylim(-5, 58.3)
    
    # Create a cleaner title
    plt.suptitle(f"{play_info['possessionTeam']} vs {play_info['defensiveTeam']} - PPPI: {pppi_score:.3f}", 
                 y=0.95, fontsize=14)
    
    # Add play details as subtitle
    # Truncate play description if it's too long
    play_desc = play_info['playDescription']
    if len(play_desc) > 50:
        play_desc = play_desc[:47] + "..."
    
    # Convert numeric down to ordinal text
    down_text = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}.get(play_info['down'], str(play_info['down']))
    
    plt.title(f"{down_text} & {play_info['yardsToGo']} - {play_desc}", 
              fontsize=10, pad=20, wrap=True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(False)
    return plt

def find_extreme_pppi_plays(play_features_with_pppi, n_plays=5):
    """
    Find plays with extreme (high/low) PPPI scores.
    
    Args:
        play_features_with_pppi: DataFrame containing plays with PPPI scores
        n_plays: Number of plays to return for each extreme
    
    Returns:
        tuple of (low_pppi_plays, high_pppi_plays)
    """
    # Sort by PPPI
    sorted_plays = play_features_with_pppi.sort_values('pppi')
    
    # Get lowest and highest PPPI plays
    low_pppi_plays = sorted_plays.head(n_plays)
    high_pppi_plays = sorted_plays.tail(n_plays)
    
    return low_pppi_plays, high_pppi_plays