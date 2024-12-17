"""
Data loading and processing functions for PPPI.
"""

import pandas as pd
import numpy as np
from pathlib import Path

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