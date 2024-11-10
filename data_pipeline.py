import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_tracking_data(weeks=range(1, 10)):
    """
    Load and combine tracking data from multiple weeks.
    """
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
    """
    Load games, plays, players, and player play data.
    """
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
    """
    Filter for relevant passing plays (excluding spikes, kneels, etc.).
    """
    passing_plays = plays_df[
        (plays_df['isDropback'] == 1) &  # Only dropback passes
        (plays_df['qbSpike'] == 0) &     # Exclude spikes
        (plays_df['qbKneel'] == 0) &     # Exclude kneels
        (plays_df['playNullifiedByPenalty'] == 'N')  # Exclude nullified plays
    ].copy()
    
    print(f"\nFiltered to {len(passing_plays)} valid passing plays")
    return passing_plays

def process_tracking_data(tracking_df, passing_plays):
    """
    Process tracking data to get pre-snap information.
    """
    # Filter for pre-snap frames in passing plays
    pre_snap = tracking_df[
        (tracking_df['frameType'] == 'BEFORE_SNAP') &
        (tracking_df['gameId'].isin(passing_plays['gameId'])) &
        (tracking_df['playId'].isin(passing_plays['playId']))
    ].copy()
    
    # Sort by game, play, frame
    pre_snap.sort_values(['gameId', 'playId', 'frameId'], inplace=True)
    
    print(f"\nPre-snap frames for passing plays: {len(pre_snap)}")
    return pre_snap

def get_defensive_pressure_data(player_play_df, passing_plays):
    """
    Get pressure-related data for defensive players.
    """
    pressure_data = player_play_df[
        (player_play_df['gameId'].isin(passing_plays['gameId'])) &
        (player_play_df['playId'].isin(passing_plays['playId']))
    ][['gameId', 'playId', 'nflId', 'causedPressure', 
       'timeToPressureAsPassRusher', 'getOffTimeAsPassRusher']].copy()
    
    print(f"\nPressure events found: {pressure_data['causedPressure'].sum()}")
    return pressure_data

def create_analysis_dataset():
    """
    Main function to create the initial analysis dataset.
    """
    print("Loading data...")
    tracking_df = load_tracking_data()
    static_data = load_static_data()
    
    if static_data is None:
        return None
    
    games_df, plays_df, players_df, player_play_df = static_data
    
    print("\nProcessing data...")
    # Filter for passing plays
    passing_plays = filter_passing_plays(plays_df)
    
    # Get pre-snap tracking data
    pre_snap_data = process_tracking_data(tracking_df, passing_plays)
    
    # Get pressure data
    pressure_data = get_defensive_pressure_data(player_play_df, passing_plays)
    
    # Create initial analysis dataset
    analysis_data = {
        'passing_plays': passing_plays,
        'pre_snap_data': pre_snap_data,
        'pressure_data': pressure_data,
        'players': players_df
    }
    
    return analysis_data

def main():
    """
    Main execution function with basic analysis.
    """
    # Create dataset
    data = create_analysis_dataset()
    if data is None:
        return
    
    # Basic summary statistics
    print("\nDataset Summary:")
    print(f"Total passing plays: {len(data['passing_plays'])}")
    print(f"Unique games: {data['passing_plays']['gameId'].nunique()}")
    
    # Pressure statistics
    total_pressures = data['pressure_data']['causedPressure'].sum()
    total_plays = len(data['passing_plays'])
    pressure_rate = (total_pressures / total_plays) * 100
    
    print(f"\nPressure Statistics:")
    print(f"Total pressures: {total_pressures}")
    print(f"Pressure rate: {pressure_rate:.1f}%")
    
    # Example of pre-snap frame distribution
    frames_per_play = data['pre_snap_data'].groupby(['gameId', 'playId'])['frameId'].count()
    print("\nPre-snap frames per play:")
    print(frames_per_play.describe())

if __name__ == "__main__":
    main()