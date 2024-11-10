import pandas as pd

def check_dataframes():
    """
    Check the structure of our dataframes before analysis
    """
    # Load the data
    tracking_df = pd.read_csv('nfl-big-data-bowl-2025/tracking_week_1.csv')
    plays_df = pd.read_csv('nfl-big-data-bowl-2025/plays.csv')
    players_df = pd.read_csv('nfl-big-data-bowl-2025/players.csv')
    
    # Print column names for each dataframe
    print("\nTracking Data Columns:")
    print(tracking_df.columns.tolist())
    
    print("\nPlays Data Columns:")
    print(plays_df.columns.tolist())
    
    print("\nPlayers Data Columns:")
    print(players_df.columns.tolist())
    
    # Print a sample of each
    print("\nTracking Data Sample (first row):")
    print(tracking_df.iloc[0])
    
    print("\nPlayers Data Sample (first row):")
    print(players_df.iloc[0])

if __name__ == "__main__":
    check_dataframes()