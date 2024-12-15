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
    
    # Add pre-snap movement features
    movement_features = defensive_alignments.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'defenders_in_motion': sum(x['s'] > 0.1),
            'defenders_accelerating': sum(x['a'] > 0.1),
            'defenders_facing_backfield': sum(abs(x['o'] - 90) < 45),
            'defenders_moving_forward': sum(abs(x['dir'] - 90) < 45)
        })
    ).reset_index()
    
    # Add defensive formation features
    formation_features = defensive_alignments.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'defensive_asymmetry': abs(sum(x['distance_from_ball_y'] < 26.65) - 
                                     sum(x['distance_from_ball_y'] >= 26.65)),
            'max_gap_between_defenders': max(np.diff(sorted(x['distance_from_ball_y']))) if len(x) > 1 else 0,
            'defenders_outside_numbers': sum((x['distance_from_ball_y'] <= 13.3) | (x['distance_from_ball_y'] >= 40)),
            'defenders_between_hashes': sum(x['distance_from_ball_y'].between(23.3, 29.9))
        })
    ).reset_index()
    
    # Add situational features
    situational_features = plays_df[['gameId', 'playId', 'down', 'yardsToGo', 
                                    'absoluteYardlineNumber', 'preSnapHomeScore',
                                    'preSnapVisitorScore']].copy()
    
    situational_features['score_differential'] = abs(
        situational_features['preSnapHomeScore'] - 
        situational_features['preSnapVisitorScore']
    )
    
    # Merge all features
    play_features = (play_features
        .merge(movement_features, on=['gameId', 'playId'])
        .merge(formation_features, on=['gameId', 'playId'])
        .merge(situational_features, on=['gameId', 'playId'])
        .merge(ol_features, on=['gameId', 'playId'])
    )
    
    # Add categorical variables
    categorical_features = ['offenseFormation', 'dropbackType', 'pff_passCoverage', 'pff_manZone']
    for feature in categorical_features:
        if feature in plays_df.columns:
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
    
    # Analyze results
    enhanced_features_with_pppi['pppi_quartile'] = pd.qcut(
        enhanced_features_with_pppi['pppi'], 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )
    pressure_by_quartile = enhanced_features_with_pppi.groupby('pppi_quartile')['causedPressure'].agg(['mean', 'count'])
    print("\nPressure Rate by PPPI Quartile:")
    print(pressure_by_quartile)
    
    plt.show()
    return enhanced_features_with_pppi, enhanced_model, enhanced_importance

if __name__ == "__main__":
    (enhanced_features_with_pppi, enhanced_model, enhanced_importance) = main()