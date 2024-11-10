import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def analyze_defensive_alignment(pre_snap_data, pressure_data, plays_df, players_df):
    """
    Analyze defensive alignments and their relationship to pressure success
    """
    print("\nStarting defensive alignment analysis...")
    
    # Print frame counts per play before processing
    frames_per_play = pre_snap_data.groupby(['gameId', 'playId']).size()
    print("\nFrames per play before processing:")
    print(frames_per_play.describe())
    
    # Get a sample play to debug frame selection
    sample_game_id = pre_snap_data['gameId'].iloc[0]
    sample_play_id = pre_snap_data['playId'].iloc[0]
    
    print(f"\nDebugging frames for game {sample_game_id}, play {sample_play_id}:")
    sample_frames = pre_snap_data[
        (pre_snap_data['gameId'] == sample_game_id) & 
        (pre_snap_data['playId'] == sample_play_id)
    ]
    print("\nNumber of frames:", len(sample_frames))
    print("\nUnique players in this play:", sample_frames['displayName'].nunique())
    
    # Get all players in the last frame of each play
    last_frames = []
    for (game_id, play_id), play_data in pre_snap_data.groupby(['gameId', 'playId']):
        max_frame = play_data['frameId'].max()
        last_frame = play_data[play_data['frameId'] == max_frame]
        last_frames.append(last_frame)
    
    last_frames_df = pd.concat(last_frames, ignore_index=True)
    
    # Print diagnostic information about the frames
    print("\nFrame Analysis:")
    print(f"Total plays processed: {len(last_frames_df['playId'].unique())}")
    print(f"Average players per play: {len(last_frames_df) / len(last_frames_df['playId'].unique()):.1f}")
    
    # Merge with plays data
    merged_data = last_frames_df.merge(
        plays_df[['gameId', 'playId', 'defensiveTeam', 'absoluteYardlineNumber', 'possessionTeam']], 
        on=['gameId', 'playId']
    )
    
    # Print sample play information
    sample_play = merged_data[merged_data['playId'] == merged_data['playId'].iloc[0]]
    print(f"\nExample Play Analysis:")
    print(f"Play ID: {sample_play['playId'].iloc[0]}")
    print(f"Defensive Team: {sample_play['defensiveTeam'].iloc[0]}")
    print(f"Possession Team: {sample_play['possessionTeam'].iloc[0]}")
    print(f"\nAll players in this play:")
    print(sample_play[['club', 'displayName', 'x', 'y']].to_string())
    
    # Identify defensive players
    defensive_alignments = merged_data[merged_data['club'] == merged_data['defensiveTeam']].copy()
    
    # Remove the football
    defensive_alignments = defensive_alignments[~defensive_alignments['displayName'].str.contains('football', case=False, na=False)]
    
    # Print defensive player counts
    players_per_play = defensive_alignments.groupby(['gameId', 'playId']).size()
    print("\nDefensive players per play:")
    print(players_per_play.describe())
    print("\nDistribution of defensive players per play:")
    print(players_per_play.value_counts().sort_index())
    
    if players_per_play.mean() < 8:
        print("\nWARNING: Too few defensive players identified!")
        return None, None
    
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
    

        # Analyze player positioning
    print("\nDefensive Player Positioning Analysis:")
    print("\nDistance from LOS distribution:")
    print(defensive_alignments['distance_from_los'].describe())
    
    print("\nY-coordinate (width) distribution:")
    print(defensive_alignments['y'].describe())
    
    # Analyze player positions relative to the box
    print("\nPlayers by distance from LOS:")
    dist_ranges = [(-999, -5), (-5, 0), (0, 5), (5, 999)]
    for start, end in dist_ranges:
        count = len(defensive_alignments[defensive_alignments['distance_from_los'].between(start, end)])
        print(f"{start} to {end} yards: {count} players")
    
    print("\nPlayers by field width position:")
    width_ranges = [(0, field_center-tackle_box_width), 
                   (field_center-tackle_box_width, field_center+tackle_box_width),
                   (field_center+tackle_box_width, field_width)]
    for start, end in width_ranges:
        count = len(defensive_alignments[defensive_alignments['y'].between(start, end)])
        print(f"{start:.1f} to {end:.1f} yards: {count} players")
    
    # Sample of player positions from one play
    print("\nSample play defensive alignment:")
    sample_play = defensive_alignments[defensive_alignments['playId'] == defensive_alignments['playId'].iloc[0]]
    print("\nDefensive players sorted by distance from LOS:")
    print(sample_play[['displayName', 'distance_from_los', 'y', 'in_box']].sort_values('distance_from_los'))
    
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
    
    # Analyze box counts
    print("\nBox Count Distribution:")
    print(play_summary['defenders_in_box'].value_counts().sort_index())

    # Final statistics
    print("\nFinal Statistics:")
    print(f"Total plays analyzed: {len(play_summary)}")
    print(f"Average defenders in box: {play_summary['defenders_in_box'].mean():.1f}")
    print(f"Average defensive players: {play_summary['defensive_players'].mean():.1f}")
    print(f"Plays with pressure: {play_summary['causedPressure'].sum()}")
    
    return play_summary, defensive_alignments


def plot_alignment_insights(play_summary):
    """
    Create visualizations for defensive alignment analysis
    """
    if play_summary is None:
        print("No data available for plotting")
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distribution of defenders in box
    sns.boxplot(data=play_summary, y='defenders_in_box', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Defenders in Box')
    axes[0,0].set_ylabel('Number of Defenders')
    
    # Plot 2: Pressure rate by number of defenders in box
    pressure_by_box = play_summary.groupby('defenders_in_box')['causedPressure'].mean()
    pressure_by_box.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Pressure Rate by Defenders in Box')
    axes[0,1].set_ylabel('Pressure Rate')
    
    # Plot 3: Defensive spread impact
    sns.scatterplot(data=play_summary, x='defensive_spread', y='depth_variation',
                   hue='causedPressure', ax=axes[1,0])
    axes[1,0].set_title('Defensive Spread vs Depth Variation')
    
    # Plot 4: Box defenders vs Depth
    sns.scatterplot(data=play_summary, x='defenders_in_box', y='avg_depth',
                   hue='causedPressure', ax=axes[1,1])
    axes[1,1].set_title('Defenders in Box vs Average Depth')
    
    plt.tight_layout()
    return fig

def main():
    # Load and process all data
    print("Loading tracking data...")
    tracking_df = load_tracking_data(weeks=[1])  # Start with week 1
    
    print("\nLoading static data...")
    static_data = load_static_data()
    if static_data is None:
        return
    
    games_df, plays_df, players_df, player_play_df = static_data
    
    # Add some data validation
    print("\nData Validation:")
    print(f"Unique teams in tracking data: {tracking_df['club'].nunique()}")
    print(f"Team abbreviations in tracking data: {sorted(tracking_df['club'].unique())}")
    print(f"Defensive teams in plays data: {sorted(plays_df['defensiveTeam'].unique())}")
    
    # Filter for passing plays
    passing_plays = filter_passing_plays(plays_df)
    
    # Get pre-snap tracking data
    pre_snap_data = process_tracking_data(tracking_df, passing_plays)
    
    # Get pressure data
    pressure_data = get_defensive_pressure_data(player_play_df, passing_plays)
    
    # Analyze alignments
    print("\nAnalyzing defensive alignments...")
    play_summary, defensive_alignments = analyze_defensive_alignment(
        pre_snap_data, 
        pressure_data, 
        passing_plays,
        players_df  # Added players_df parameter
    )
    
    if play_summary is None:
        print("Analysis failed due to data issues.")
        return None, None, None
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = plot_alignment_insights(play_summary)
    
    # Print key findings
    print("\nDefensive Alignment Analysis:")
    print(f"Average defenders in box: {play_summary['defenders_in_box'].mean():.2f}")
    print(f"Average defensive players on field: {play_summary['defensive_players'].mean():.2f}")
    print(f"Correlation between defenders in box and pressure: {play_summary['defenders_in_box'].corr(play_summary['causedPressure']):.3f}")
    
    plt.show()
    return play_summary, defensive_alignments, fig

if __name__ == "__main__":
    play_summary, defensive_alignments, fig = main()







    