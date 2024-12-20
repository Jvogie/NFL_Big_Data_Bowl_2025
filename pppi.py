import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
import shap
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTETomek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings('ignore')

def load_tracking_data(weeks=range(1, 10), data_dir='nfl-big-data-bowl-2025'):
    """Load and combine tracking data from multiple weeks."""
    tracking_frames = []
    
    for week in weeks:
        try:
            file_path = Path(data_dir) / f'tracking_week_{week}.csv'
            df = pd.read_csv(file_path)
            tracking_frames.append(df)
            print(f"Loaded week {week} tracking data: {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    if not tracking_frames:
        raise FileNotFoundError("No tracking data files were found")
        
    tracking_df = pd.concat(tracking_frames, ignore_index=True)
    print(f"\nTotal tracking data rows: {len(tracking_df)}")
    return tracking_df

def load_static_data(data_dir='nfl-big-data-bowl-2025'):
    """Load games, plays, players, and player play data."""
    try:
        # Create Path objects for each file
        games_path = Path(data_dir) / 'games.csv'
        plays_path = Path(data_dir) / 'plays.csv'
        players_path = Path(data_dir) / 'players.csv'
        player_play_path = Path(data_dir) / 'player_play.csv'
        
        # Load each file
        games_df = pd.read_csv(games_path)
        plays_df = pd.read_csv(plays_path)
        players_df = pd.read_csv(players_path)
        player_play_df = pd.read_csv(player_play_path)
        
        print(f"Loaded static data:")
        print(f"Games: {len(games_df)} rows")
        print(f"Plays: {len(plays_df)} rows")
        print(f"Players: {len(players_df)} rows")
        print(f"Player plays: {len(player_play_df)} rows")
        
        return games_df, plays_df, players_df, player_play_df
    
    except FileNotFoundError as e:
        print(f"Error loading static data: {e}")
        print("\nPlease ensure the following files exist in the '{data_dir}' directory:")
        print("- games.csv")
        print("- plays.csv")
        print("- players.csv")
        print("- player_play.csv")
        return None

def filter_passing_plays(plays_df):
    """Filter for relevant passing plays."""
    passing_plays = plays_df[
        (plays_df['isDropback'] == 1) &
        (plays_df['qbSpike'] == 0) &
        (plays_df['qbKneel'] == 0) &
        (plays_df['playNullifiedByPenalty'] == 'N')
    ].copy()
    
    print(f"\nFiltered to {len(passing_plays)} valid passing plays")
    return passing_plays

def process_tracking_data(tracking_df, passing_plays):
    """Process tracking data to get pre-snap information."""
    # Create a set of valid (gameId, playId) combinations for faster lookup
    valid_plays = set(zip(passing_plays['gameId'], passing_plays['playId']))
    
    # Filter tracking data to only include valid plays
    pre_snap = tracking_df[
        tracking_df['frameType'] == 'BEFORE_SNAP'
    ].copy()
    
    # Create (gameId, playId) tuples for filtering
    pre_snap['play_key'] = list(zip(pre_snap['gameId'], pre_snap['playId']))
    
    # Filter to only valid plays
    pre_snap = pre_snap[pre_snap['play_key'].isin(valid_plays)]
    
    # Remove the temporary key column
    pre_snap.drop('play_key', axis=1, inplace=True)
    
    # Debug prints
    print(f"\nSample of tracking data:")
    print(tracking_df[['gameId', 'playId', 'frameType', 'nflId']].head())
    print(f"\nUnique frameTypes: {tracking_df['frameType'].unique()}")
    print(f"\nNumber of rows with null nflId (should be ball): {tracking_df['nflId'].isna().sum()}")
    
    # Sort by game, play, and frame
    pre_snap.sort_values(['gameId', 'playId', 'frameId'], inplace=True)
    
    print(f"\nPre-snap frames for passing plays: {len(pre_snap)}")
    print(f"Unique plays in tracking data: {len(pre_snap.groupby(['gameId', 'playId']))}")
    print(f"Unique plays in plays.csv: {len(passing_plays)}")
    
    return pre_snap

def get_defensive_pressure_data(player_play_df, passing_plays):
    """Get pressure-related data for defensive players."""
    pressure_data = player_play_df[
        (player_play_df['gameId'].isin(passing_plays['gameId'])) &
        (player_play_df['playId'].isin(passing_plays['playId']))
    ][['gameId', 'playId', 'nflId', 'causedPressure', 
       'timeToPressureAsPassRusher', 'getOffTimeAsPassRusher']].copy()
    
    print(f"\nPressure events found: {pressure_data['causedPressure'].sum()}")
    return pressure_data

def analyze_defensive_alignment(pre_snap_data, pressure_data, plays_df, players_df):
    """Analyze defensive alignments relative to the ball position."""
    print("\nStarting defensive alignment analysis...")
    
    # Get all players in the last frame before snap
    last_frames = []
    plays_without_ball = []
    play_count = 0
    
    for (game_id, play_id), play_data in pre_snap_data.groupby(['gameId', 'playId']):
        # Get the last frame before snap
        max_frame = play_data['frameId'].max()
        last_frame = play_data[play_data['frameId'] == max_frame]
        
        # Get ball position for this play
        ball_data = last_frame[last_frame['nflId'].isna()]
        
        if len(ball_data) == 0:
            print(f"Warning: No ball position found for game {game_id}, play {play_id}")
            plays_without_ball.append((game_id, play_id))
            continue
        
        # Get defensive team for this play
        play_info = plays_df[
            (plays_df['gameId'] == game_id) & 
            (plays_df['playId'] == play_id)
        ]
        
        if len(play_info) == 0:
            print(f"Warning: No play information found for game {game_id}, play {play_id}")
            continue
            
        defensive_team = play_info['defensiveTeam'].iloc[0]
        
        # Debug print for first few plays only
        if play_count < 3:  # Show details for first 3 plays
            ball_pos = ball_data.iloc[0]
            ball_x, ball_y = ball_pos['x'], ball_pos['y']
            print(f"\nBall position for game {game_id}, play {play_id}:")
            print(f"x: {ball_x:.2f}, y: {ball_y:.2f}")
            
            # Debug print for defensive positions
            def_players = last_frame[last_frame['club'] == defensive_team]
            
            print(f"Defensive players positions relative to ball:")
            for _, player in def_players.iterrows():
                rel_x = player['x'] - ball_x
                rel_y = player['y'] - ball_y
                print(f"Player {player['displayName']}: x_rel: {rel_x:.2f}, y_rel: {rel_y:.2f}")
        
        # Add ball position to all rows for this play
        ball_pos = ball_data.iloc[0]
        last_frame['ball_x'] = ball_pos['x']
        last_frame['ball_y'] = ball_pos['y']
        
        last_frames.append(last_frame)
        play_count += 1
    
    print(f"\nProcessed {play_count} plays total")
    
    if plays_without_ball:
        print(f"Total plays without ball position: {len(plays_without_ball)}")
        
    if not last_frames:
        raise ValueError("No valid plays found with ball position data")
        
    last_frames_df = pd.concat(last_frames, ignore_index=True)
    
    # Merge with plays data
    merged_data = last_frames_df.merge(
        plays_df[['gameId', 'playId', 'defensiveTeam', 'possessionTeam']], 
        on=['gameId', 'playId']
    )
    
    # Identify defensive players
    defensive_alignments = merged_data[
        (merged_data['club'] == merged_data['defensiveTeam']) & 
        (~merged_data['displayName'].str.contains('football', case=False, na=False))
    ].copy()
    
    # Calculate distances relative to ball position
    defensive_alignments['distance_from_ball_x'] = np.where(
        defensive_alignments['playDirection'] == 'right',
        defensive_alignments['x'] - defensive_alignments['ball_x'],
        defensive_alignments['ball_x'] - defensive_alignments['x']
    )
    
    defensive_alignments['distance_from_ball_y'] = np.abs(
        defensive_alignments['y'] - defensive_alignments['ball_y']
    )
    
    # Define box area (typically 5 yards from LOS and between the tackles)
    box_depth = 5  # yards behind LOS
    tackle_box_width = 4  # yards from center on each side
    
    # Define defensive position categories
    defensive_alignments['on_line'] = (
        defensive_alignments['distance_from_ball_x'].between(-1, 1)
    ).astype(int)
    
    defensive_alignments['in_box'] = (
        (defensive_alignments['distance_from_ball_x'].between(-box_depth, 2)) &  # Include players slightly ahead of ball
        (defensive_alignments['distance_from_ball_y'] <= tackle_box_width)
    ).astype(int)
    
    defensive_alignments['edge_position'] = (
        (defensive_alignments['distance_from_ball_x'].between(-2, 2)) &
        (defensive_alignments['distance_from_ball_y'].between(tackle_box_width, tackle_box_width + 2))
    ).astype(int)
    
    # Group by play for summary statistics
    play_summary = defensive_alignments.groupby(['gameId', 'playId']).agg({
        'in_box': 'sum',
        'on_line': 'sum',
        'edge_position': 'sum',
        'distance_from_ball_x': ['mean', 'std', 'min', 'max'],
        'distance_from_ball_y': ['mean', 'std', 'min', 'max'],
        'defensiveTeam': 'first'
    }).reset_index()
    
    # Flatten columns
    play_summary.columns = [
        'gameId', 'playId', 'defenders_in_box', 'defenders_on_line', 'edge_defenders',
        'avg_depth', 'depth_variation', 'closest_defender_depth', 'deepest_defender_depth',
        'avg_width', 'width_variation', 'nearest_defender_width', 'widest_defender_width',
        'defensiveTeam'
    ]
    
    # Add pressure information
    pressure_by_play = pressure_data.groupby(['gameId', 'playId'])['causedPressure'].max().reset_index()
    play_summary = play_summary.merge(pressure_by_play, on=['gameId', 'playId'], how='left')
    play_summary['causedPressure'] = play_summary['causedPressure'].fillna(0)
    
    return play_summary, defensive_alignments

def create_basic_pressure_features(defensive_alignments, play_summary, plays_df):
    """Create basic features for pressure prediction with proper categorical handling."""
    # Group defensive alignments by play
    play_features = defensive_alignments.groupby(['gameId', 'playId']).agg({
        'distance_from_los': ['mean', 'std', 'min', 'max'],
        'y': ['mean', 'std', 'min', 'max'],
        'in_box': 'sum'
    }).reset_index()
    
    # Flatten multi-level columns
    play_features.columns = ['gameId', 'playId', 
                           'avg_depth', 'depth_std', 'closest_defender', 'deepest_defender',
                           'avg_width', 'width_std', 'left_most', 'right_most',
                           'box_defenders']
    
    # Add numeric features first
    play_features = play_features.merge(
        plays_df[['gameId', 'playId', 'playAction']], 
        on=['gameId', 'playId']
    )
    
    # Convert playAction to numeric
    play_features['playAction'] = play_features['playAction'].astype(int)
    
    # Handle categorical features separately
    categorical_features = ['offenseFormation', 'dropbackType']
    for feature in categorical_features:
        # Create dummy variables
        dummies = pd.get_dummies(plays_df[['gameId', 'playId', feature]], 
                               columns=[feature], 
                               prefix=feature)
        play_features = play_features.merge(dummies, on=['gameId', 'playId'])
    
    # Add pressure outcome
    play_features = play_features.merge(
        play_summary[['gameId', 'playId', 'causedPressure']], 
        on=['gameId', 'playId']
    )
    
    return play_features

def create_enhanced_pressure_features(defensive_alignments, play_summary, plays_df, player_play_df, ol_features):
    """Create enhanced features including offensive line metrics."""
    # Base defensive features
    play_features = defensive_alignments.groupby(['gameId', 'playId']).agg({
        'distance_from_ball_x': ['mean', 'std', 'min', 'max'],
        'distance_from_ball_y': ['mean', 'std', 'min', 'max'],
        'in_box': 'sum',
        'on_line': 'sum',
        'edge_position': 'sum',
        's': ['mean', 'max'],  # Speed metrics
        'a': ['mean', 'max'],  # Acceleration metrics
        'o': ['mean', 'std'],  # Orientation metrics
        'dir': ['mean', 'std']  # Direction metrics
    }).reset_index()
    
    # Flatten multi-level columns
    play_features.columns = ['gameId', 'playId', 
                           'avg_depth', 'depth_std', 'closest_defender', 'deepest_defender',
                           'avg_width', 'width_std', 'nearest_width', 'widest_width',
                           'box_defenders', 'line_defenders', 'edge_defenders',
                           'avg_speed', 'max_speed',
                           'avg_accel', 'max_accel',
                           'avg_orientation', 'orientation_std',
                           'avg_direction', 'direction_std']
    
    # Add play situation features
    play_features = play_features.merge(
        plays_df[['gameId', 'playId', 'down', 'yardsToGo', 'absoluteYardlineNumber']], 
        on=['gameId', 'playId']
    )
    
    # Add non-pressure matchup features
    matchup_features = player_play_df.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'num_pass_rushers': sum(x['wasInitialPassRusher'] == 1),
            'num_blockers': x['blockedPlayerNFLId1'].notna().sum(),
            'rusher_to_blocker_ratio': sum(x['wasInitialPassRusher'] == 1) / 
                                      (x['blockedPlayerNFLId1'].notna().sum() + 1e-6)
        })
    ).reset_index()
    
    # Add defensive clustering features
    clustering_features = defensive_alignments.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'min_defender_spacing': min(np.diff(sorted(x['distance_from_ball_y']))) if len(x) > 1 else 0,
            'avg_defender_spacing': np.mean(np.diff(sorted(x['distance_from_ball_y']))) if len(x) > 1 else 0,
            'front_7_depth_variance': x[x['distance_from_ball_x'] <= 7]['distance_from_ball_x'].var(),
            'front_7_width_variance': x[x['distance_from_ball_x'] <= 7]['distance_from_ball_y'].var(),
            'rushers_with_speed': sum((x['s'] > 2) & (x['distance_from_ball_x'] <= 5)),
            'rushers_accelerating': sum((x['a'] > 1) & (x['distance_from_ball_x'] <= 5))
        })
    ).reset_index()
    
    # Merge features
    play_features = (play_features
        .merge(matchup_features, on=['gameId', 'playId'])
        .merge(clustering_features, on=['gameId', 'playId'])
        .merge(ol_features, on=['gameId', 'playId'])
    )
    
    # Create feature interactions
    play_features = create_feature_interactions(play_features)
    
    # Add target variable last and only for training
    if 'causedPressure' in play_summary.columns:
        play_features = play_features.merge(
            play_summary[['gameId', 'playId', 'causedPressure']], 
            on=['gameId', 'playId']
        )
    
    return play_features

def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple models including stacked ensemble."""
    # Add stacked model to our models dictionary
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=2, random_state=42),
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
        'Stacked Ensemble': create_stacked_model()
    }
    
    results = {}
    feature_importance_df = pd.DataFrame()
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.3f}")
        
        # Get feature importance (skip for Stacked Ensemble)
        if name != 'Stacked Ensemble':
            importance = model.feature_importances_
            
            # Store feature importance
            model_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance,
                'Model': name
            })
            feature_importance_df = pd.concat([feature_importance_df, model_importance])
        
        # Store results
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    # Only plot feature importance if we have data
    if not feature_importance_df.empty:
        # Plot feature importance comparison
        plt.figure(figsize=(12, 6))
        top_features = (feature_importance_df.groupby('Feature')['Importance']
                       .mean()
                       .sort_values(ascending=False)
                       .head(10)
                       .index)
        
        comparison_df = feature_importance_df[feature_importance_df['Feature'].isin(top_features)]
        
        sns.barplot(data=comparison_df, x='Importance', y='Feature', hue='Model')
        plt.title('Feature Importance Comparison Across Models')
        plt.tight_layout()
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    return results, feature_importance_df

def analyze_feature_importance(model, X_train, X_test, y_train, y_test, feature_names):
    """Analyze feature importance using multiple methods."""
    importance_results = {}
    
    # 1. Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    importance_results['permutation'] = perm_importance_df
    
    # 2. SHAP Values (for tree-based models)
    if hasattr(model, 'predict_proba'):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):  # For multi-class output
                shap_values = shap_values[1]  # Take positive class
                
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
    
    # 3. Recursive Feature Elimination
    rfe = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=20
    )
    
    rfe.fit(X_train, y_train)
    rfe_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    importance_results['rfe'] = rfe_importance_df
    
    # Plot combined importance analysis
    plt.figure(figsize=(12, 6))
    
    # Get top 20 features from permutation importance
    top_features = perm_importance_df.head(20)['Feature']
    
    # Create comparison plot
    comparison_data = []
    for feature in top_features:
        feature_data = {
            'Feature': feature,
            'Permutation': perm_importance_df[perm_importance_df['Feature'] == feature]['Importance'].iloc[0],
            'SHAP': shap_importance_df[shap_importance_df['Feature'] == feature]['Importance'].iloc[0] if 'shap' in importance_results else 0,
            'RFE_Selected': rfe_importance_df[rfe_importance_df['Feature'] == feature]['Selected'].iloc[0]
        }
        comparison_data.append(feature_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Normalize importance scores
    comparison_df['Permutation'] = comparison_df['Permutation'] / comparison_df['Permutation'].max()
    if 'shap' in importance_results:
        comparison_df['SHAP'] = comparison_df['SHAP'] / comparison_df['SHAP'].max()
    
    # Plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=comparison_df, x='Permutation', y='Feature')
    plt.title('Permutation Importance')
    
    if 'shap' in importance_results:
        plt.subplot(1, 2, 2)
        sns.barplot(data=comparison_df, x='SHAP', y='Feature')
        plt.title('SHAP Importance')
    
    plt.tight_layout()
    
    # Print feature importance summary
    print("\nTop 10 Most Important Features (Permutation Importance):")
    print(perm_importance_df.head(10))
    
    if 'shap' in importance_results:
        print("\nTop 10 Most Important Features (SHAP):")
        print(shap_importance_df.head(10))
    
    print("\nSelected Features by RFE:")
    print(rfe_importance_df[rfe_importance_df['Selected']].sort_values('Ranking'))
    
    return importance_results

def build_pressure_model(play_features):
    """Build and evaluate multiple pressure prediction models with advanced techniques."""
    # Select features for model
    feature_columns = [col for col in play_features.columns 
                      if col not in ['gameId', 'playId', 'causedPressure']]
    
    X = play_features[feature_columns]
    y = play_features['causedPressure']
    
    # Get enhanced cross-validation strategy
    cv = get_enhanced_cv()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values first
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Use SMOTETomek for better balanced data
    smotetomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train_scaled, y_train)
    
    # Initialize models with calibration
    models = {
        'Random Forest': CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                class_weight='balanced_subsample',
                random_state=42
            ),
            cv=cv,
            method='sigmoid'
        ),
        'XGBoost': CalibratedClassifierCV(
            get_refined_xgboost(),
            cv=cv,
            method='sigmoid'
        ),
        'LightGBM': CalibratedClassifierCV(
            get_optimized_lgbm(),
            cv=cv,
            method='sigmoid'
        ),
        'CatBoost': CalibratedClassifierCV(
            get_optimized_catboost(),
            cv=cv,
            method='sigmoid'
        ),
        'Voting Ensemble': get_voting_ensemble()
    }
    
    # Train neural network separately
    nn_model = NeuralNetworkWrapper()
    nn_model.fit(X_train_balanced, y_train_balanced)
    nn_pred_proba = nn_model.predict_proba(X_test_scaled)
    nn_pred = nn_model.predict(X_test_scaled)
    nn_roc_auc = roc_auc_score(y_test, nn_pred_proba[:, 1])
    
    print("\nNeural Network Results:")
    print("\nClassification Report:")
    print(classification_report(y_test, nn_pred))
    print(f"ROC AUC Score: {nn_roc_auc:.3f}")
    
    # Train and evaluate models with cross-validation
    results = {}
    feature_importance_df = pd.DataFrame()
    
    for name, model in models.items():
        if name != 'Neural Network':
            print(f"\n{name} Results:")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced, 
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            print(f"Cross-validation ROC AUC scores: {cv_scores}")
            print(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print(f"ROC AUC Score: {roc_auc:.3f}")
            
            # Store results
            results[name] = {
                'model': model,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'cv_scores': cv_scores
            }
    
    # Create stacked model using calibrated base models
    base_estimators = [
        (name.lower().replace(' ', '_'), model) 
        for name, model in models.items()
    ]
    
    stacked_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=8,
                max_depth=3,
                class_weight='balanced',
                
                random_state=42,
                verbose=-1
            ),
            cv=cv,
            method='sigmoid'
        ),
        cv=cv,
        n_jobs=-1
    )
    
    # Train and evaluate stacked model
    print("\nStacked Ensemble Results:")
    stacked_model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = stacked_model.predict(X_test_scaled)
    y_pred_proba = stacked_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc:.3f}")
    
    results['Stacked Ensemble'] = {
        'model': stacked_model,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Use the best performing model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    
    # Analyze feature importance for the best model
    importance_analysis = analyze_feature_importance(
        best_model,
        X_train_balanced,
        X_test_scaled,
        y_train_balanced,
        y_test,
        feature_columns
    )
    
    return best_model, scaler, feature_columns, importance_analysis

def calculate_pppi(model, scaler, feature_columns, play_features):
    """Calculate the Pre-snap Pressure Prediction Index."""
    # Select and order features
    X = play_features[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Calculate PPPI
    pppi = model.predict_proba(X_scaled)[:, 1]
    
    # Create a copy of play_features to avoid modifying the original
    play_features_with_pppi = play_features.copy()
    play_features_with_pppi['pppi'] = pppi
    
    # Print distribution statistics
    print("\nPPPI Distribution:")
    print(pd.Series(pppi).describe())
    
    return play_features_with_pppi

def analyze_offensive_line(pre_snap_data, plays_df):
    """Analyze offensive line formations and configurations."""
    # Get offensive players in last frame before snap
    last_frames = []
    for (game_id, play_id), play_data in pre_snap_data.groupby(['gameId', 'playId']):
        max_frame = play_data['frameId'].max()
        last_frame = play_data[play_data['frameId'] == max_frame]
        last_frames.append(last_frame)
    
    last_frames_df = pd.concat(last_frames, ignore_index=True)
    
    # Merge with plays data to get possession team
    merged_data = last_frames_df.merge(
        plays_df[['gameId', 'playId', 'possessionTeam', 'absoluteYardlineNumber']], 
        on=['gameId', 'playId']
    )
    
    # Identify offensive linemen (typically 5 players closest to LOS)
    offensive_players = merged_data[merged_data['club'] == merged_data['possessionTeam']].copy()
    
    # Calculate distance from LOS for each player
    offensive_players['distance_from_los'] = np.where(
        offensive_players['playDirection'] == 'right',
        offensive_players['absoluteYardlineNumber'] - offensive_players['x'],
        offensive_players['x'] - offensive_players['absoluteYardlineNumber']
    )
    
    # Group by play and get offensive line features
    play_summary = offensive_players.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'ol_width': x.nsmallest(5, 'distance_from_los')['y'].max() - 
                       x.nsmallest(5, 'distance_from_los')['y'].min(),
            'ol_depth': x.nsmallest(5, 'distance_from_los')['distance_from_los'].mean(),
            'ol_spacing': x.nsmallest(5, 'distance_from_los')['y'].diff().abs().mean(),
            'ol_balanced': (abs(x.nsmallest(5, 'distance_from_los')['y'].mean() - 26.65) < 2),
            'tight_end_attached': any((x['y'] < 13.3) | (x['y'] > 40)),
        })
    ).reset_index()
    
    return play_summary

def create_feature_interactions(play_features):
    """Create interaction features between important variables."""
    # Base interactions
    play_features['speed_depth_interaction'] = play_features['avg_speed'] * play_features['avg_depth']
    play_features['box_pressure_potential'] = play_features['box_defenders'] * play_features['avg_speed']
    play_features['edge_speed_threat'] = play_features['edge_defenders'] * play_features['max_speed']
    
    # Enhanced defensive formation features based on important metrics
    play_features['defensive_spread'] = play_features['width_std'] * play_features['depth_std']
    play_features['box_density'] = play_features['box_defenders'] / (play_features['width_std'] + 1e-6)
    
    # Refined front complexity using top features
    play_features['front_complexity'] = (
        play_features['line_defenders'] * 
        play_features['edge_defenders'] * 
        play_features['width_std'] *
        (1 + play_features['avg_orientation'] / 180)  # Add orientation impact
    )
    
    # Enhanced rusher effectiveness metric
    play_features['rusher_effectiveness'] = (
        play_features['num_pass_rushers'] * 
        play_features['avg_speed'] * 
        (1 / (play_features['avg_defender_spacing'] + 1e-6)) *
        (1 + play_features['front_complexity'] / 10)
    )
    
    # Defensive momentum and positioning
    play_features['defensive_momentum'] = (
        play_features['avg_speed'] * 
        play_features['avg_accel'] * 
        (1 / (play_features['avg_depth'] + 1)) *
        (1 + play_features['orientation_std'] / 180)  # Add orientation factor
    )
    
    # Enhanced edge pressure calculations
    play_features['edge_pressure_potential'] = (
        play_features['edge_defenders'] * 
        play_features['max_speed'] * 
        (1 / (play_features['ol_spacing'] + 1e-6)) *
        (1 + play_features['ol_width'] / 20)  # Factor in OL width
    )
    
    # Situational pressure features
    if 'down' in play_features.columns and 'yardsToGo' in play_features.columns:
        # Critical down and distance situations
        play_features['pressure_situation'] = (
            ((play_features['down'] >= 3) & (play_features['yardsToGo'] >= 7)) |  # 3rd/4th and long
            ((play_features['down'] == 2) & (play_features['yardsToGo'] >= 10))   # 2nd and very long
        ).astype(int)
        
        # Enhanced situational pressure intensity
        play_features['situational_pressure_intensity'] = (
            play_features['pressure_situation'] * 
            play_features['defensive_momentum'] * 
            (1 + play_features['down'] / 4) *  # Down weight
            (1 + play_features['yardsToGo'] / 20)  # Distance weight
        )
         # Down-specific pressure likelihood
        play_features['down_pressure'] = np.where(
            play_features['down'] >= 3,
            play_features['down'] * 1.5,  # Higher weight for 3rd/4th down
            play_features['down']
        )

        
        # Enhanced situational pressure
        play_features['situation_severity'] = (
            play_features['down_pressure'] * 
            (1 + play_features['yardsToGo'] / 10) *  # Scale by yards to go
            (1 + (play_features['num_pass_rushers'] > 4).astype(int) * 0.5)  # Extra rushers bonus
        )
    
    # Defensive spacing effectiveness
    play_features['spacing_effectiveness'] = (
        (1 / (play_features['min_defender_spacing'] + 1e-6)) * 
        play_features['avg_defender_spacing'] *
        play_features['front_complexity']
    )
    
    # OL stress metrics based on important features
    play_features['ol_pressure_index'] = (
        play_features['num_pass_rushers'] / 
        (play_features['ol_width'] + 1e-6) *
        (1 + play_features['front_complexity'] / 10)
    )
    
    # Normalize key metrics
    for col in ['rusher_effectiveness', 'spacing_effectiveness', 'ol_pressure_index']:
        if col in play_features.columns:
            play_features[col] = np.clip(
                play_features[col],
                play_features[col].quantile(0.05),
                play_features[col].quantile(0.95)
            )
    
    return play_features

def create_stacked_model():
    base_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            random_state=42
        )),
        ('xgb', get_refined_xgboost()),  # Use optimized XGBoost
        ('lgb', get_optimized_lgbm())    # Use optimized LightGBM
    ]
    
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=get_optimized_lgbm(),  # Use optimized LightGBM as final estimator
        cv=get_enhanced_cv(),  # Use enhanced CV strategy
        n_jobs=-1
    )

def get_optimized_lgbm():
    return LGBMClassifier(
        n_estimators=200,            # Reduced from 300
        learning_rate=0.05,          # Increased from 0.02
        num_leaves=31,               # Increased from 16
        max_depth=6,                 # Increased from 4
        min_child_samples=20,        # Reduced from 30
        subsample=0.8,
        colsample_bytree=0.8,
        min_split_gain=1e-3,         # Added minimum split gain
        min_child_weight=1e-3,       # Added minimum child weight
        class_weight='balanced',
        reg_alpha=0.1,               # Reduced from 0.2
        reg_lambda=0.1,              # Reduced from 0.2
        random_state=42,
        verbose=-1
    )

def create_weighted_ensemble(models, weights):
    def weighted_predict_proba(X):
        probas = np.array([model.predict_proba(X) for model in models])
        return np.average(probas, axis=0, weights=weights)
    return weighted_predict_proba

def get_refined_xgboost():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=2,
        reg_alpha=0.3,
        reg_lambda=0.3,
        random_state=42
    )

def get_enhanced_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class NeuralNetworkWrapper:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_params(self, deep=True):
        # Implement get_params for scikit-learn compatibility
        return {"model": self.model, "device": self.device}
    
    def set_params(self, **parameters):
        # Implement set_params for scikit-learn compatibility
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = NeuralNetwork(X.shape[1]).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X, y.reshape(-1, 1))
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(100):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        # Implement predict for scikit-learn compatibility
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X):
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        
        # Return probabilities for both classes
        return np.column_stack([1-preds, preds])

def get_optimized_catboost():
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        class_weights={0: 1, 1: 2},
        random_seed=42,
        verbose=False
    )

def get_voting_ensemble():
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            random_state=42
        )),
        ('xgb', get_refined_xgboost()),
        ('lgb', get_optimized_lgbm()),
        ('cat', get_optimized_catboost())
    ]
    
    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[1, 1.2, 1.2, 1.2]
    )

def main():
    # Use the default directory name where the data files are located
    data_dir = "nfl-big-data-bowl-2025"
    
    # Load data
    tracking_df = load_tracking_data(data_dir=data_dir)
    static_data = load_static_data(data_dir=data_dir)
    if static_data is None:
        return
    
    games_df, plays_df, players_df, player_play_df = static_data
    
    # Process plays and get alignments
    passing_plays = filter_passing_plays(plays_df)
    
    # Validate data consistency
    print("\nValidating data consistency...")
    print(f"Total plays in plays.csv: {len(plays_df)}")
    print(f"Passing plays after filtering: {len(passing_plays)}")
    
    tracking_plays = set(zip(tracking_df['gameId'], tracking_df['playId']))
    passing_play_keys = set(zip(passing_plays['gameId'], passing_plays['playId']))
    missing_from_tracking = passing_play_keys - tracking_plays
    
    if missing_from_tracking:
        print(f"\nWarning: {len(missing_from_tracking)} passing plays not found in tracking data")
        print("First few missing plays:")
        for game_id, play_id in list(missing_from_tracking)[:5]:
            play_info = plays_df[
                (plays_df['gameId'] == game_id) & 
                (plays_df['playId'] == play_id)
            ].iloc[0]
            print(f"Game {game_id}, Play {play_id}: {play_info['playDescription']}")
    
    # Continue with processing
    pre_snap_data = process_tracking_data(tracking_df, passing_plays)
    pressure_data = get_defensive_pressure_data(player_play_df, passing_plays)
    play_summary, defensive_alignments = analyze_defensive_alignment(
        pre_snap_data, 
        pressure_data, 
        passing_plays,
        players_df
    )
    
    # Analyze offensive line and create features
    ol_features = analyze_offensive_line(pre_snap_data, plays_df)
    enhanced_features = create_enhanced_pressure_features(
        defensive_alignments, 
        play_summary, 
        plays_df,
        player_play_df,
        ol_features
    )
    
    # Build enhanced model and calculate PPPI
    enhanced_model, enhanced_scaler, enhanced_feature_cols, enhanced_importance = build_pressure_model(enhanced_features)
    enhanced_features_with_pppi = calculate_pppi(
        enhanced_model, 
        enhanced_scaler, 
        enhanced_feature_cols, 
        enhanced_features
    )
    
    # Analyze results with handling for duplicate bin edges
    enhanced_features_with_pppi['pppi_quartile'] = pd.qcut(
        enhanced_features_with_pppi['pppi'], 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'  # Handle duplicate bin edges
    )
    
    # Analyze results
    pressure_by_quartile = enhanced_features_with_pppi.groupby('pppi_quartile')['causedPressure'].agg(['mean', 'count'])
    print("\nPressure Rate by PPPI Quartile:")
    print(pressure_by_quartile)
    
    plt.show()
    return enhanced_features_with_pppi, enhanced_model, enhanced_importance

if __name__ == "__main__":
    (enhanced_features_with_pppi, enhanced_model, enhanced_importance) = main()