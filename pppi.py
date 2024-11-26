# Areas to improve: add in offensive line formation


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
warnings.filterwarnings('ignore')

def load_tracking_data(weeks=range(1, 10)):
    """Load and combine tracking data from multiple weeks."""
    tracking_frames = []
    
    for week in weeks:
        try:
            df = pd.read_csv(f'nfl-big-data-bowl-2025/tracking_week_{week}.csv')
            tracking_frames.append(df)
            print(f"Loaded week {week} tracking data: {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: tracking_week_{week}.csv not found")
    
    tracking_df = pd.concat(tracking_frames, ignore_index=True)
    print(f"\nTotal tracking data rows: {len(tracking_df)}")
    return tracking_df

def load_static_data():
    """Load games, plays, players, and player play data."""
    try:
        games_df = pd.read_csv('nfl-big-data-bowl-2025/games.csv')
        plays_df = pd.read_csv('nfl-big-data-bowl-2025/plays.csv')
        players_df = pd.read_csv('nfl-big-data-bowl-2025/players.csv')
        player_play_df = pd.read_csv('nfl-big-data-bowl-2025/player_play.csv')
        
        print(f"Loaded static data:")
        print(f"Games: {len(games_df)} rows")
        print(f"Plays: {len(plays_df)} rows")
        print(f"Players: {len(players_df)} rows")
        print(f"Player plays: {len(player_play_df)} rows")
        
        return games_df, plays_df, players_df, player_play_df
    
    except FileNotFoundError as e:
        print(f"Error loading static data: {e}")
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
    pre_snap = tracking_df[
        (tracking_df['frameType'] == 'BEFORE_SNAP') &
        (tracking_df['gameId'].isin(passing_plays['gameId'])) &
        (tracking_df['playId'].isin(passing_plays['playId']))
    ].copy()
    
    pre_snap.sort_values(['gameId', 'playId', 'frameId'], inplace=True)
    print(f"\nPre-snap frames for passing plays: {len(pre_snap)}")
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
    """Analyze defensive alignments and their relationship to pressure success."""
    print("\nStarting defensive alignment analysis...")
    
    # Get all players in the last frame of each play
    last_frames = []
    for (game_id, play_id), play_data in pre_snap_data.groupby(['gameId', 'playId']):
        max_frame = play_data['frameId'].max()
        last_frame = play_data[play_data['frameId'] == max_frame]
        last_frames.append(last_frame)
    
    last_frames_df = pd.concat(last_frames, ignore_index=True)
    
    # Merge with plays data
    merged_data = last_frames_df.merge(
        plays_df[['gameId', 'playId', 'defensiveTeam', 'absoluteYardlineNumber', 'possessionTeam']], 
        on=['gameId', 'playId']
    )
    
    # Identify defensive players
    defensive_alignments = merged_data[merged_data['club'] == merged_data['defensiveTeam']].copy()
    defensive_alignments = defensive_alignments[~defensive_alignments['displayName'].str.contains('football', case=False, na=False)]
    
    # Calculate distance from LOS
    defensive_alignments['distance_from_los'] = np.where(
        defensive_alignments['playDirection'] == 'right',
        defensive_alignments['absoluteYardlineNumber'] - defensive_alignments['x'],
        defensive_alignments['x'] - defensive_alignments['absoluteYardlineNumber']
    )
    
    # Define box area
    field_width = 53.3
    field_center = field_width / 2
    tackle_box_width = 8
    box_depth = 5
    
    defensive_alignments['in_box'] = (
        (defensive_alignments['distance_from_los'].between(-box_depth, box_depth)) &
        (defensive_alignments['y'].between(field_center - tackle_box_width, field_center + tackle_box_width))
    ).astype(int)
    
    # Group by play
    play_summary = defensive_alignments.groupby(['gameId', 'playId']).agg({
        'in_box': 'sum',
        'distance_from_los': ['mean', 'std'],
        'y': ['std', 'count'],
        'defensiveTeam': 'first'
    }).reset_index()
    
    # Flatten columns
    play_summary.columns = ['gameId', 'playId', 'defenders_in_box', 'avg_depth', 
                          'depth_variation', 'defensive_spread', 'defensive_players',
                          'defensiveTeam']
    
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

def create_enhanced_pressure_features(defensive_alignments, play_summary, plays_df, player_play_df):
    """Create enhanced features for pressure prediction with proper categorical handling."""
    # Original positioning features
    play_features = defensive_alignments.groupby(['gameId', 'playId']).agg({
        'distance_from_los': ['mean', 'std', 'min', 'max'],
        'y': ['mean', 'std', 'min', 'max'],
        'in_box': 'sum'
    }).reset_index()
    
    # Flatten multi-level columns first
    play_features.columns = ['gameId', 'playId', 
                           'avg_depth', 'depth_std', 'closest_defender', 'deepest_defender',
                           'avg_width', 'width_std', 'left_most', 'right_most',
                           'box_defenders']
    
    # Add new defensive formation features
    def_formation_features = defensive_alignments.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'edge_defenders': sum((x['distance_from_los'].abs() <= 2) & 
                                (x['y'].between(13.3, 20) | x['y'].between(33.3, 40))),
            'interior_defenders': sum((x['distance_from_los'].abs() <= 2) & 
                                    x['y'].between(20, 33.3)),
            'second_level_defenders': sum((x['distance_from_los'].between(2, 5))),
            'left_side_defenders': sum(x['y'] < 26.65),
            'right_side_defenders': sum(x['y'] >= 26.65),
            'defenders_within_3yards': sum(x['distance_from_los'].abs() <= 3),
            'max_defender_cluster': max(sum(abs(x['y'] - y) < 3) for y in x['y']),
            'defenders_outside_numbers': sum((x['y'] <= 13.3) | (x['y'] >= 40)),
            'defenders_between_hashes': sum(x['y'].between(23.3, 29.9)),
            'max_gap_between_defenders': max(np.diff(sorted(x['y']))) if len(x) > 1 else 0
        })
    ).reset_index()
    
    # Merge new features
    play_features = play_features.merge(def_formation_features, on=['gameId', 'playId'])
    
    # Add only numeric offensive context initially
    play_features = play_features.merge(
        plays_df[['gameId', 'playId', 'playAction']], 
        on=['gameId', 'playId']
    )
    
    # Now add categorical variables that need to be encoded
    categorical_features = ['offenseFormation', 'dropbackType', 'pff_passCoverage', 'pff_manZone']
    for feature in categorical_features:
        # Create dummy variables for each categorical feature
        dummies = pd.get_dummies(plays_df[['gameId', 'playId', feature]], 
                               columns=[feature], 
                               prefix=feature)
        play_features = play_features.merge(dummies, on=['gameId', 'playId'])
    
    # Add pressure outcome
    play_features = play_features.merge(
        play_summary[['gameId', 'playId', 'causedPressure']], 
        on=['gameId', 'playId']
    )
    
    # Calculate relative features
    play_features['edge_to_interior_ratio'] = (play_features['edge_defenders'] / (play_features['interior_defenders'] + 0.1))
    play_features['second_level_ratio'] = (play_features['second_level_defenders'] / (play_features['defenders_within_3yards'] + 0.1))
    play_features['defensive_imbalance'] = abs(play_features['left_side_defenders'] - play_features['right_side_defenders'])
    play_features['center_presence'] = play_features['defenders_between_hashes'] / (play_features['box_defenders'] + 0.1)
    
    # Convert playAction to numeric
    play_features['playAction'] = play_features['playAction'].astype(int)
    
    return play_features

def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train and evaluate multiple models on the same data
    """
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=2
        ),
        'LightGBM': LGBMClassifier(
            # Basic parameters
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            
            # Class imbalance handling
            class_weight='balanced',
            
            # Tree structure parameters
            max_depth=6,
            num_leaves=31,  # 2^(max_depth) - 1
            min_child_samples=20,  # Minimum number of data needed in a leaf
            
            # Split finding parameters
            min_split_gain=0.1,  # Minimum gain to make a split
            min_child_weight=1e-3,  # Minimum sum of instance weight needed in a child
            
            # Regularization parameters
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            
            # Performance parameters
            force_row_wise=True,  # Remove overhead warning
            verbose=-1,  # Reduce verbosity
            
            # Additional parameters for better splits
            feature_fraction=0.8,  # Use 80% of features in each iteration
            bagging_fraction=0.8,  # Use 80% of data in each iteration
            bagging_freq=5,  # Perform bagging every 5 iterations
        )
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
        
        # Get feature importance
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

def build_pressure_model(play_features):
    """Build and evaluate multiple pressure prediction models."""
    # Select features for model
    feature_columns = [col for col in play_features.columns 
                      if col not in ['gameId', 'playId', 'causedPressure']]
    
    X = play_features[feature_columns]
    y = play_features['causedPressure']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    results, feature_importance = train_multiple_models(
        X_train_scaled, 
        X_test_scaled, 
        y_train, 
        y_test, 
        feature_columns
    )
    
    # Use the best performing model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    
    return best_model, scaler, feature_columns, feature_importance


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

def main():
    # Load and process data
    print("Loading tracking data...")
    tracking_df = load_tracking_data()
    
    print("\nLoading static data...")
    static_data = load_static_data()
    if static_data is None:
        return
    
    games_df, plays_df, players_df, player_play_df = static_data
    
    # Data validation
    print("\nData Validation:")
    print(f"Unique teams in tracking data: {tracking_df['club'].nunique()}")
    print(f"Team abbreviations in tracking data: {sorted(tracking_df['club'].unique())}")
    print(f"Defensive teams in plays data: {sorted(plays_df['defensiveTeam'].unique())}")
    
    # Process plays
    passing_plays = filter_passing_plays(plays_df)
    pre_snap_data = process_tracking_data(tracking_df, passing_plays)
    pressure_data = get_defensive_pressure_data(player_play_df, passing_plays)
    
    # Analyze alignments
    play_summary, defensive_alignments = analyze_defensive_alignment(
        pre_snap_data, 
        pressure_data, 
        passing_plays,
        players_df
    )
    
    # Create features
    print("\nCreating basic pressure prediction features...")
    basic_features = create_basic_pressure_features(defensive_alignments, play_summary, plays_df)
    
    print("\nCreating enhanced pressure prediction features...")
    enhanced_features = create_enhanced_pressure_features(
        defensive_alignments, 
        play_summary, 
        plays_df,
        player_play_df
    )
    
    # Build models
    print("\nBuilding basic pressure prediction model...")
    basic_model, basic_scaler, basic_feature_cols, basic_importance = build_pressure_model(basic_features)
    
    print("\nBuilding enhanced pressure prediction model...")
    enhanced_model, enhanced_scaler, enhanced_feature_cols, enhanced_importance = build_pressure_model(enhanced_features)
    
    # Calculate PPPI
    print("\nCalculating Basic PPPI...")
    basic_features_with_pppi = calculate_pppi(basic_model, basic_scaler, basic_feature_cols, basic_features)
    
    print("\nCalculating Enhanced PPPI...")
    enhanced_features_with_pppi = calculate_pppi(enhanced_model, enhanced_scaler, enhanced_feature_cols, enhanced_features)
    
    # Compare model performances
    print("\nModel Comparison:")
    print("\nBasic Model Pressure Rate by PPPI Quartile:")
    basic_features_with_pppi['pppi_quartile'] = pd.qcut(basic_features_with_pppi['pppi'], 
                                                       q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    basic_pressure_by_quartile = basic_features_with_pppi.groupby('pppi_quartile')['causedPressure'].agg(['mean', 'count'])
    print(basic_pressure_by_quartile)
    
    print("\nEnhanced Model Pressure Rate by PPPI Quartile:")
    enhanced_features_with_pppi['pppi_quartile'] = pd.qcut(enhanced_features_with_pppi['pppi'], 
                                                          q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    enhanced_pressure_by_quartile = enhanced_features_with_pppi.groupby('pppi_quartile')['causedPressure'].agg(['mean', 'count'])
    print(enhanced_pressure_by_quartile)
    
    plt.show()
    return (basic_features_with_pppi, basic_model, basic_importance,
            enhanced_features_with_pppi, enhanced_model, enhanced_importance)

if __name__ == "__main__":
    (basic_features_with_pppi, basic_model, basic_importance,
     enhanced_features_with_pppi, enhanced_model, enhanced_importance) = main()