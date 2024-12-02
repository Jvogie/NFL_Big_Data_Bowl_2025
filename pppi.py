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

def analyze_football_data(defensive_alignments):
    """Analyze availability of football tracking data."""
    print("\nFootball Data Analysis:")
    
    # Basic data checks
    print("\nChecking for football in data:")
    print(f"Values in displayName column: {defensive_alignments['displayName'].unique()}")
    
    # Get football data
    football_data = defensive_alignments[defensive_alignments['displayName'] == 'football']
    print(f"\nTotal football data rows: {len(football_data)}")
    
    # Count plays
    total_plays = defensive_alignments['playId'].nunique()
    plays_with_football = football_data['playId'].nunique()
    
    print(f"\nSummary:")
    print(f"Total unique plays: {total_plays}")
    print(f"Plays with football data: {plays_with_football}")
    print(f"Percentage of plays with football: {(plays_with_football/total_plays)*100:.2f}%")
    
    # Sample of football data
    print("\nSample of football tracking data:")
    sample_play = football_data.groupby('playId').first().sample(n=1, random_state=42).index[0]
    sample_data = football_data[football_data['playId'] == sample_play].sort_values('frameId')
    print(f"\nPlay ID: {sample_play}")
    print(sample_data[['frameId', 'frameType', 'x', 'y', 'event']].head())
    
    return plays_with_football, football_data

def create_oline_features(defensive_alignments, play_summary, plays_df):
    """Create features based on defensive alignment relative to ball position."""
    def calculate_oline_matchups(group):
        """Calculate defensive alignment features for a single play using ball position."""
        # Check for football position
        football_data = group[group['displayName'] == 'football']
        if len(football_data) == 0:
            # Use field center as fallback
            ball_position = 26.65
        else:
            ball_position = football_data['y'].iloc[0]
        
        # Only look at defenders
        defenders = group[
            (group['displayName'] != 'football') & 
            (group['distance_from_los'].abs() <= 3)
        ]
        
        if len(defenders) == 0:
            # Return default values if no defenders
            return pd.Series({
                'defenders_over_LT': 0,
                'defenders_over_LG': 0,
                'defenders_over_C': 0,
                'defenders_over_RG': 0,
                'defenders_over_RT': 0,
                'closest_defender_to_LT': 3,
                'closest_defender_to_LG': 3,
                'closest_defender_to_C': 3,
                'closest_defender_to_RG': 3,
                'closest_defender_to_RT': 3,
                'defenders_in_A_gap_left': 0,
                'defenders_in_A_gap_right': 0,
                'defenders_in_B_gap_left': 0,
                'defenders_in_B_gap_right': 0,
                'closest_defender_in_A_gap_left': 3,
                'closest_defender_in_A_gap_right': 3,
                'closest_defender_in_B_gap_left': 3,
                'closest_defender_in_B_gap_right': 3,
                'defensive_line_balance': 0,
                'overloaded_side': 0,
                'defenders_head_up': 0,
                'wide_defenders_left': 0,
                'wide_defenders_right': 0
            })
        
        # Define O-line positions relative to ball position
        oline_positions = {
            'LT': ball_position - 4,
            'LG': ball_position - 2,
            'C': ball_position,
            'RG': ball_position + 2,
            'RT': ball_position + 4
        }
        
        features = {}
        
        # Calculate features
        for position, y_coord in oline_positions.items():
            position_defenders = defenders[
                (defenders['y'] >= y_coord - 1) & 
                (defenders['y'] <= y_coord + 1)
            ]
            
            features[f'defenders_over_{position}'] = len(position_defenders)
            features[f'closest_defender_to_{position}'] = (
                position_defenders['distance_from_los'].abs().min() 
                if len(position_defenders) > 0 else 3
            )
        
        # Add gap features
        gaps = {
            'A_gap_left': (oline_positions['C'], oline_positions['LG']),
            'A_gap_right': (oline_positions['C'], oline_positions['RG']),
            'B_gap_left': (oline_positions['LG'], oline_positions['LT']),
            'B_gap_right': (oline_positions['RG'], oline_positions['RT'])
        }
        
        for gap_name, (pos1, pos2) in gaps.items():
            gap_defenders = defenders[
                (defenders['y'] >= min(pos1, pos2)) & 
                (defenders['y'] <= max(pos1, pos2))
            ]
            
            features[f'defenders_in_{gap_name}'] = len(gap_defenders)
            features[f'closest_defender_in_{gap_name}'] = (
                gap_defenders['distance_from_los'].abs().min() 
                if len(gap_defenders) > 0 else 3
            )
        
        # Additional features
        features.update({
            'defensive_line_balance': abs(
                sum(defenders['y'] < ball_position) - 
                sum(defenders['y'] > ball_position)
            ),
            'overloaded_side': max(
                sum(defenders['y'] < ball_position),
                sum(defenders['y'] > ball_position)
            ),
            'defenders_head_up': sum(abs(defenders['y'] - ball_position) <= 0.5),
            'wide_defenders_left': sum(defenders['y'] <= ball_position - 8),
            'wide_defenders_right': sum(defenders['y'] >= ball_position + 8)
        })
        
        return pd.Series(features)
    
    # Group by play and calculate features
    return defensive_alignments.groupby(['gameId', 'playId']).apply(calculate_oline_matchups).reset_index()

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
    """Create enhanced features for pressure prediction."""

    print("\nAnalyzing football tracking data availability...")
    plays_with_football, plays_missing_football = analyze_football_data(defensive_alignments)
    
    def calculate_box_features(group):
        """Calculate box features relative to ball position."""
        # Check for football position
        football_data = group[group['displayName'] == 'football']
        if len(football_data) == 0:
            # print(f"Warning: No football found for gameId={group['gameId'].iloc[0]}, playId={group['playId'].iloc[0]}")
            # Use middle of field as fallback
            ball_position = 26.65
        else:
            ball_position = football_data['y'].iloc[0]

        # Get only defensive players (exclude football)
        defenders = group[group['displayName'] != 'football']
        
        # Define box dimensions
        box_width = 8  # yards total (4 yards each side of ball)
        box_depth = 5  # yards from LOS
        
        # Calculate in_box relative to ball position
        in_box = (
            (defenders['distance_from_los'].abs() <= box_depth) &  # Within 5 yards of LOS
            (abs(defenders['y'] - ball_position) <= box_width/2)   # Within box width centered on ball
        ).astype(int)
        
        features = {
            'defenders_in_box': sum(in_box),
            'ball_position': ball_position  # Add this for debugging
        }
        
        # Only calculate other stats if we have defenders
        if len(defenders) > 0:
            features.update({
                'avg_depth': defenders['distance_from_los'].mean(),
                'depth_std': defenders['distance_from_los'].std(),
                'closest_defender': defenders['distance_from_los'].abs().min(),
                'deepest_defender': defenders['distance_from_los'].abs().max(),
                'avg_width': defenders['y'].mean(),
                'width_std': defenders['y'].std(),
                'left_most': defenders['y'].min(),
                'right_most': defenders['y'].max(),
            })
        else:
            # Default values if no defenders
            features.update({
                'avg_depth': 0,
                'depth_std': 0,
                'closest_defender': 0,
                'deepest_defender': 0,
                'avg_width': ball_position,
                'width_std': 0,
                'left_most': ball_position,
                'right_most': ball_position,
            })
        
        return pd.Series(features)

    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Total plays: {defensive_alignments['playId'].nunique()}")
    print(f"Plays with football: {defensive_alignments[defensive_alignments['displayName'] == 'football']['playId'].nunique()}")
    
    # Sample a few plays to check data
    sample_plays = defensive_alignments.groupby('playId').first().sample(n=5, random_state=42)
    print("\nSample plays:")
    for _, play in sample_plays.iterrows():
        play_data = defensive_alignments[defensive_alignments['playId'] == play.name]
        print(f"\nPlayId: {play.name}")
        print(f"Total frames: {len(play_data)}")
        print(f"Has football: {'football' in play_data['displayName'].values}")
        print(f"Unique players: {play_data['displayName'].nunique()}")

    # Calculate basic defensive alignment features
    play_features = defensive_alignments.groupby(['gameId', 'playId']).apply(calculate_box_features).reset_index()
    
    # Add O-line alignment features
    oline_features = create_oline_features(defensive_alignments, play_summary, plays_df)
    play_features = play_features.merge(oline_features, on=['gameId', 'playId'])
    
    def calculate_formation_features(group):
        """Calculate formation features relative to ball position."""
        # Get ball position
        football_data = group[group['displayName'] == 'football']
        if len(football_data) == 0:
            ball_position = 26.65  # Fallback to middle
        else:
            ball_position = football_data['y'].iloc[0]
            
        defenders = group[group['displayName'] != 'football']
        
        if len(defenders) == 0:
            return pd.Series({
                'edge_defenders': 0,
                'interior_defenders': 0,
                'second_level_defenders': 0,
                'left_side_defenders': 0,
                'right_side_defenders': 0,
                'defenders_within_3yards': 0,
                'max_defender_cluster': 0,
                'defenders_outside_numbers': 0,
                'defenders_between_hashes': 0,
                'max_gap_between_defenders': 0
            })
        
        return pd.Series({
            'edge_defenders': sum((defenders['distance_from_los'].abs() <= 2) & 
                                ((defenders['y'] <= ball_position - 6) | 
                                 (defenders['y'] >= ball_position + 6))),
            'interior_defenders': sum((defenders['distance_from_los'].abs() <= 2) & 
                                    (abs(defenders['y'] - ball_position) <= 6)),
            'second_level_defenders': sum(defenders['distance_from_los'].between(2, 5)),
            'left_side_defenders': sum(defenders['y'] < ball_position),
            'right_side_defenders': sum(defenders['y'] >= ball_position),
            'defenders_within_3yards': sum(defenders['distance_from_los'].abs() <= 3),
            'max_defender_cluster': max(sum(abs(defenders['y'] - y) < 3) 
                                     for y in defenders['y']),
            'defenders_outside_numbers': sum((defenders['y'] <= ball_position - 13) | 
                                          (defenders['y'] >= ball_position + 13)),
            'defenders_between_hashes': sum(abs(defenders['y'] - ball_position) <= 3),
            'max_gap_between_defenders': max(np.diff(sorted(defenders['y']))) 
                                       if len(defenders) > 1 else 0
        })
    
    # Add formation features
    formation_features = defensive_alignments.groupby(['gameId', 'playId']).apply(calculate_formation_features).reset_index()
    play_features = play_features.merge(formation_features, on=['gameId', 'playId'])
    
    # Add offensive context
    play_features = play_features.merge(
        plays_df[['gameId', 'playId', 'offenseFormation', 'dropbackType', 
                 'playAction', 'pff_passCoverage', 'pff_manZone']], 
        on=['gameId', 'playId']
    )
    
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
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
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
    """Build and evaluate multiple pressure prediction models with proper categorical handling."""
    # Identify categorical columns
    categorical_columns = [
        'offenseFormation', 
        'dropbackType', 
        'pff_passCoverage', 
        'pff_manZone'
    ]
    
    # Convert playAction to numeric if it isn't already
    play_features['playAction'] = play_features['playAction'].astype(int)
    
    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(
        play_features, 
        columns=categorical_columns,
        prefix=categorical_columns
    )
    
    # Select features for model (exclude non-feature columns)
    feature_columns = [col for col in df_encoded.columns 
                      if col not in ['gameId', 'playId', 'causedPressure']]
    
    X = df_encoded[feature_columns]
    y = df_encoded['causedPressure']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print feature information
    print("\nFeature Information:")
    print(f"Total features: {len(feature_columns)}")
    print("\nFeature types:")
    for col in X.columns:
        print(f"{col}: {X[col].dtype}")
    
    # Train and evaluate models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            min_split_gain=0.1,
            min_child_weight=1e-3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            force_row_wise=True,
            verbose=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5
        )
    }
    
    results = {}
    feature_importance_df = pd.DataFrame()
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.3f}")
        
        # Get feature importance
        importance = model.feature_importances_
            
        # Store feature importance
        model_importance = pd.DataFrame({
            'Feature': feature_columns,
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
    
    # Use best performing model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    print(f"\nBest performing model: {best_model_name}")
    
    return best_model, scaler, feature_columns, feature_importance_df

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
    # print("\nData Validation:")
    # print(f"Unique teams in tracking data: {tracking_df['club'].nunique()}")
    # print(f"Team abbreviations in tracking data: {sorted(tracking_df['club'].unique())}")
    # print(f"Defensive teams in plays data: {sorted(plays_df['defensiveTeam'].unique())}")
    
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