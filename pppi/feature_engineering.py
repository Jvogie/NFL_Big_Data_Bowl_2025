"""
Feature engineering and processing functions for PPPI.
"""

import numpy as np
import pandas as pd

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
        last_frame = play_data[play_data['frameId'] == max_frame].copy()  # Create explicit copy
        
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
        
        # Add ball position to all rows for this play
        ball_pos = ball_data.iloc[0]
        last_frame.loc[:, 'ball_x'] = ball_pos['x']
        last_frame.loc[:, 'ball_y'] = ball_pos['y']
        
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
    
    # Define box area and defensive position categories
    box_depth = 5  # yards behind LOS
    tackle_box_width = 4  # yards from center on each side
    
    defensive_alignments['on_line'] = (
        defensive_alignments['distance_from_ball_x'].between(-1, 1)
    ).astype(int)
    
    defensive_alignments['in_box'] = (
        (defensive_alignments['distance_from_ball_x'].between(-box_depth, 2)) &
        (defensive_alignments['distance_from_ball_y'] <= tackle_box_width)
    ).astype(int)
    
    defensive_alignments['edge_position'] = (
        (defensive_alignments['distance_from_ball_x'].between(-2, 2)) &
        (defensive_alignments['distance_from_ball_y'].between(tackle_box_width, tackle_box_width + 2))
    ).astype(int)
    
    return defensive_alignments

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
    
    # Identify offensive linemen
    offensive_players = merged_data[merged_data['club'] == merged_data['possessionTeam']].copy()
    
    # Calculate distance from LOS
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
            'tight_end_attached': any((x['y'] < 13.3) | (x['y'] > 40))
        })
    ).reset_index()
    
    return play_summary

def create_enhanced_pressure_features(defensive_alignments, pressure_data, plays_df, player_play_df, ol_features):
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
    
    # Add pressure data
    pressure_summary = pressure_data.groupby(['gameId', 'playId'])['causedPressure'].max().reset_index()
    play_features = play_features.merge(pressure_summary, on=['gameId', 'playId'], how='left')
    
    # Add non-pressure matchup features
    matchup_features = player_play_df.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'num_pass_rushers': sum(x['wasInitialPassRusher'] == 1),
            'num_blockers': x['blockedPlayerNFLId1'].notna().sum(),
            'rusher_to_blocker_ratio': sum(x['wasInitialPassRusher'] == 1) / 
                                      (x['blockedPlayerNFLId1'].notna().sum() + 1e-6)
        })
    ).reset_index()
    
    # Add defensive clustering features with NaN handling
    clustering_features = defensive_alignments.groupby(['gameId', 'playId']).apply(
        lambda x: pd.Series({
            'min_defender_spacing': min(np.diff(sorted(x['distance_from_ball_y']))) if len(x) > 1 else 0,
            'avg_defender_spacing': np.mean(np.diff(sorted(x['distance_from_ball_y']))) if len(x) > 1 else 0,
            'front_7_depth_variance': (
                x[x['distance_from_ball_x'] <= 7]['distance_from_ball_x'].var()
                if len(x[x['distance_from_ball_x'] <= 7]) > 1 else 0
            ),
            'front_7_width_variance': (
                x[x['distance_from_ball_x'] <= 7]['distance_from_ball_y'].var()
                if len(x[x['distance_from_ball_x'] <= 7]) > 1 else 0
            ),
            'rushers_with_speed': sum((x['s'] > 2) & (x['distance_from_ball_x'] <= 5)),
            'rushers_accelerating': sum((x['a'] > 1) & (x['distance_from_ball_x'] <= 5))
        })
    ).reset_index()
    
    # Merge all features
    play_features = (play_features
        .merge(matchup_features, on=['gameId', 'playId'])
        .merge(clustering_features, on=['gameId', 'playId'])
        .merge(ol_features, on=['gameId', 'playId'])
    )
    
    # Create feature interactions
    play_features = create_feature_interactions(play_features)
    
    # Fill any missing values in causedPressure with 0
    play_features['causedPressure'] = play_features['causedPressure'].fillna(0)
    
    return play_features

def create_feature_interactions(play_features):
    """Create interaction features between important variables."""
    # Base interactions
    play_features['speed_depth_interaction'] = play_features['avg_speed'] * play_features['avg_depth']
    play_features['box_pressure_potential'] = play_features['box_defenders'] * play_features['avg_speed']
    play_features['edge_speed_threat'] = play_features['edge_defenders'] * play_features['max_speed']
    
    # Enhanced defensive formation features
    play_features['defensive_spread'] = play_features['width_std'] * play_features['depth_std']
    play_features['box_density'] = play_features['box_defenders'] / (play_features['width_std'] + 1e-6)
    
    # Front complexity
    play_features['front_complexity'] = (
        play_features['line_defenders'] * 
        play_features['edge_defenders'] * 
        play_features['width_std'] *
        (1 + play_features['avg_orientation'] / 180)
    )
    
    # Enhanced rusher effectiveness
    play_features['rusher_effectiveness'] = (
        play_features['num_pass_rushers'] * 
        play_features['avg_speed'] * 
        (1 / (play_features['avg_defender_spacing'] + 1e-6)) *
        (1 + play_features['front_complexity'] / 10)
    )
    
    # Situational pressure features
    if 'down' in play_features.columns and 'yardsToGo' in play_features.columns:
        play_features['down_pressure'] = np.where(
            play_features['down'] >= 3,
            play_features['down'] * 1.5,
            play_features['down']
        )
        
        play_features['situation_severity'] = (
            play_features['down_pressure'] * 
            (1 + play_features['yardsToGo'] / 10) *
            (1 + (play_features['num_pass_rushers'] > 4).astype(int) * 0.5)
        )
    
    # Normalize key metrics
    for col in ['rusher_effectiveness', 'front_complexity', 'situation_severity']:
        if col in play_features.columns:
            play_features[col] = np.clip(
                play_features[col],
                play_features[col].quantile(0.05),
                play_features[col].quantile(0.95)
            )
    
    return play_features 