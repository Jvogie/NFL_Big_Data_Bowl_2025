"""
Main script for running the PPPI analysis.
"""

import matplotlib.pyplot as plt
from pppi.data_loading import (
    load_tracking_data,
    load_static_data,
    filter_passing_plays,
    process_tracking_data,
    get_defensive_pressure_data
)
from pppi.feature_engineering import (
    analyze_defensive_alignment,
    analyze_offensive_line,
    create_enhanced_pressure_features
)
from pppi.evaluation import (
    build_pressure_model,
    calculate_pppi
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
    
    # Process tracking data
    pre_snap_data = process_tracking_data(tracking_df, passing_plays)
    pressure_data = get_defensive_pressure_data(player_play_df, passing_plays)
    
    # Create features
    defensive_alignments = analyze_defensive_alignment(
        pre_snap_data, 
        pressure_data, 
        passing_plays,
        players_df
    )
    
    ol_features = analyze_offensive_line(pre_snap_data, plays_df)
    
    enhanced_features = create_enhanced_pressure_features(
        defensive_alignments, 
        pressure_data,
        plays_df,
        player_play_df,
        ol_features
    )
    
    # Build model and calculate PPPI
    model, scaler, feature_cols, importance = build_pressure_model(enhanced_features)
    features_with_pppi = calculate_pppi(
        model, 
        scaler, 
        feature_cols, 
        enhanced_features
    )
    
    plt.show()
    return features_with_pppi, model, importance

if __name__ == "__main__":
    features_with_pppi, model, importance = main() 