import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from db.db import get_maps
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
DATASET_CACHE_PATH = CACHE_DIR / 'processed_dataset.pkl'

def load_player_stats() -> pd.DataFrame:
    try:
        df = pd.read_csv('./player_stats_complete.csv')
        
        # Convert percentage columns to decimals
        percentage_cols = ['accuracy', 'headshot_percentage']
        for col in percentage_cols:
            df[col] = df[col].str.rstrip('%').astype(float) / 100
            
        # Check for duplicate player names
        duplicate_names = df['name'].duplicated(keep=False)
        if duplicate_names.any():
            print("Warning: Found duplicate player names:")
            print(df[duplicate_names]['name'].unique())
            print("Using first occurrence of each player")
            df = df.drop_duplicates(subset=['name'], keep='first')
            
        df.set_index('name', inplace=True)
        return df
        
    except FileNotFoundError:
        print("Error: player_stats_complete.csv file not found")
        return pd.DataFrame()

def load_or_create_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Load dataset from cache if available, otherwise create and cache it"""
    if not force_refresh and DATASET_CACHE_PATH.exists():
        print("Loading dataset from cache...")
        try:
            with open(DATASET_CACHE_PATH, 'rb') as f:
                df = pickle.load(f)
            print("Dataset loaded from cache successfully")
            return df
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            print("Creating new dataset...")
    
    # Create new dataset
    maps_data = get_maps()
    print(f"Retrieved {len(maps_data)} maps from database")
    
    # Debug: Check the structure of the first few maps
    print("\nDebug: Sample of first 3 maps from database:")
    for i, map_data in enumerate(maps_data[:3]):
        print(f"\nMap {i+1}:")
        print(f"  Map name: {map_data.get('name', 'unknown')}")
        print(f"  Map ID: {map_data.get('map_id', 'unknown')}")
        print(f"  Round: {map_data.get('round', 'unknown')}")
        print(f"  Total rounds: {map_data.get('total_rounds', 'unknown')}")
        print(f"  Teams: {len(map_data.get('teams', []))} teams")
        for j, team in enumerate(map_data.get('teams', [])):
            print(f"    Team {j+1}: {team.get('team_name', 'unknown')} - {len(team.get('players', []))} players")
    
    player_stats = load_player_stats()
    print(f"Loaded {len(player_stats)} player stats")
    
    df = create_dataset(maps_data, player_stats)
    
    # Cache the dataset
    print("Caching dataset...")
    try:
        with open(DATASET_CACHE_PATH, 'wb') as f:
            pickle.dump(df, f)
        print("Dataset cached successfully")
    except Exception as e:
        print(f"Error caching dataset: {str(e)}")
    
    return df

def filter_maps(maps_data: List[Dict], player_stats: pd.DataFrame) -> List[Dict]:
    """Filter maps to only include those where all players exist in player_stats"""
    filtered_maps = []
    
    for map_data in maps_data:
        all_players_exist = True
        
        # Check all players in both teams
        for team in map_data['teams']:
            for player in team['players']:
                if player['name'] not in player_stats.index:
                    all_players_exist = False
                    break
            if not all_players_exist:
                break
                
        if all_players_exist:
            filtered_maps.append(map_data)
            
    print(f"Filtered from {len(maps_data)} to {len(filtered_maps)} maps")
    return filtered_maps

def is_valid_numeric_data(data_list: List) -> bool:
    """Check if all elements in a list are valid numeric values"""
    try:
        # Try to convert all elements to float
        numeric_data = [float(x) for x in data_list]
        # Check for any invalid values (negative kills, etc)
        if any(x < -1 for x in numeric_data):  # allowing -1 in case it's used as a special value
            return False
        return True
    except (ValueError, TypeError):
        return False

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset"""
    
    # Existing features
    df['high_rating_diff'] = df['rating_diff'].abs() > df['rating_diff'].std()
    df['squared_team_rating_advantage'] = (df['ally_rating'] - df['enemy_rating'])**2
    df['kill_consistency'] = df.groupby('name')['kills_y'].transform('std')
    df['opening_kill_ratio'] = df['opening_kills'] / (df['opening_deaths'] + 1)
    df['high_opening_kill_ratio'] = df['opening_kill_ratio'] > 1.5
    df['entry_fragger_score'] = df['opening_kills'] * df['trades'] / (df['deaths'] + 1)
    df['kd_ratio'] = df['kills'] / np.maximum(df['deaths'], 0.1)


    df['kill_to_enemy_kill_ratio'] = df['kills'] / np.maximum(df['enemy_kills'], 1)

    df['average_rounds_ratio'] = df['ally_avg_rounds_per_game'] / df['enemy_avg_rounds_per_game']
    df['average_rounds_difference'] = df['ally_avg_rounds_per_game'] - df['enemy_avg_rounds_per_game']
    df['average_rounds_scaled_by_score'] = df['ally_avg_rounds_per_game'] / df['squared_team_rating_advantage']
    df['average_rounds_scaled_by_score_difference'] = (df['ally_avg_rounds_per_game']* df['ally_rating'] - df['enemy_avg_rounds_per_game']* df['enemy_rating']).abs()
    
    return df

def create_dataset(maps_data: List[Dict], player_stats: pd.DataFrame) -> pd.DataFrame:
    # Filter maps first
    maps_data = filter_maps(maps_data, player_stats)
    
    # Print some debug information about map names and rounds
    print("\nDebug: Sample of first 5 maps from the database:")
    for map_data in maps_data[:5]:
        print(f"\nMap data:")
        print(f"  Map name: {map_data.get('map_name', 'unknown')}")  # Changed from map_name to name
        print(f"  Map ID: {map_data.get('map_id', 'unknown')}")
        print(f"  Round: {map_data.get('round', 'unknown')}")
        print(f"  Total rounds: {map_data.get('total_rounds', 'unknown')}")
    
    # First pass - calculate average rounds per game for each player
    player_rounds = {}
    for map_data in maps_data:
        total_rounds = map_data.get('total_rounds', 0)
        if not total_rounds:  # Skip if total_rounds is missing or 0
            continue
            
        # Add rounds data for all players in the match
        for team in map_data['teams']:
            for player in team['players']:
                player_name = player['name']
                if player_name not in player_rounds:
                    player_rounds[player_name] = {'total_rounds': 0, 'num_games': 0}
                player_rounds[player_name]['total_rounds'] += total_rounds
                player_rounds[player_name]['num_games'] += 1
    
    # Calculate averages and add to player_stats
    avg_rounds = {}
    for player, data in player_rounds.items():
        if data['num_games'] > 0:
            avg_rounds[player] = data['total_rounds'] / data['num_games']
    
    # Add avg_rounds to player_stats
    player_stats['avg_rounds_per_game'] = pd.Series(avg_rounds)
    
    # Create empty lists to store data
    rows = []
    stat_cols = [col for col in player_stats.columns]
    
    output_cols = (
        ['name', 'map_name', 'round'] + ['kills_y', 'total_rounds_y'] + stat_cols +  # Added map_name
        [f'ally_{col}' for col in stat_cols] +
        [f'enemy_{col}' for col in stat_cols] + 
        ['rating_diff']
    )
    
    df = pd.DataFrame(columns=output_cols)
    print("number of maps: ", len(maps_data))
    all_rows = []
    for map in maps_data:
        try:
            # Extract player kills and total rounds
            total_rounds = map.get('total_rounds', 0)
            round_number = map.get('round', 0)  # Get the round number
            map_name = map.get('map_name', 'unknown')  # Changed from map_name to name
            if not total_rounds:  # Skip if total_rounds is missing or 0
                continue
                
            t1_player_names = [player['name'] for player in map['teams'][0]['players']]
            t1_player_kills = [player['kills'] for player in map['teams'][0]['players']]
            t2_player_names = [player['name'] for player in map['teams'][1]['players']]
            t2_player_kills = [player['kills'] for player in map['teams'][1]['players']]
            
            # Validate numeric data
            if not (is_valid_numeric_data(t1_player_kills) and is_valid_numeric_data(t2_player_kills)):
                print(f"Skipping map due to invalid kill data: T1 kills: {t1_player_kills}, T2 kills: {t2_player_kills}")
                continue
                
            # Calculate team averages
            t1_averages = player_stats.loc[t1_player_names][stat_cols].mean()
            t2_averages = player_stats.loc[t2_player_names][stat_cols].mean()

            # Process team 1 players
            for i, player in enumerate(t1_player_names):
                player_row = player_stats.loc[player]
                row = [player, map_name, round_number, float(t1_player_kills[i]), float(total_rounds)] + list(player_row[stat_cols].values)            
                ally_stats = player_stats.loc[t1_player_names].drop(player)[stat_cols].mean()
                enemy_stats = t2_averages
                row += list(ally_stats.values) + list(enemy_stats.values) + [player_row['rating'] - t2_averages['rating']]
                all_rows.append(row)
                
            # Process team 2 players
            for i, player in enumerate(t2_player_names):
                player_row = player_stats.loc[player]
                row = [player, map_name, round_number, float(t2_player_kills[i]), float(total_rounds)] + list(player_row[stat_cols].values)
                ally_stats = player_stats.loc[t2_player_names].drop(player)[stat_cols].mean()
                enemy_stats = t1_averages
                row += list(ally_stats.values) + list(enemy_stats.values) + [player_row['rating'] - t1_averages['rating']]
                all_rows.append(row)
                
        except (ValueError, TypeError) as e:
            print(f"Skipping map due to error: {str(e)}")
            continue
    
    df = pd.DataFrame(all_rows, columns=output_cols)
    
    # Add engineered features
    df = add_engineered_features(df)
    
    print("number of rows: ", len(df))
    return df

def main():
    # Load or create dataset
    df = load_or_create_dataset(force_refresh=True)
    print("Dataset shape:", df.shape)

if __name__ == "__main__":
    main()
