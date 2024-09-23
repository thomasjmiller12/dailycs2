import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db import get_maps

def split_maps(maps, train_ratio=0.7):
    random.shuffle(maps)
    split_index = int(len(maps) * train_ratio)
    return maps[:split_index], maps[split_index:]

def calculate_player_averages(maps):
    player_totals = defaultdict(lambda: {'kills': 0, 'maps': 0})
    for map_data in maps:
        for team in map_data['teams']:
            for player in team['players']:
                player_totals[player['name']]['kills'] += player['kills']
                player_totals[player['name']]['maps'] += 1
    
    player_averages = {
        player: data['kills'] / data['maps']
        for player, data in player_totals.items()
    }
    return player_averages

def predict_kills(player_averages, map_data):
    predictions = []
    actual_kills = []
    for team in map_data['teams']:
        for player in team['players']:
            predictions.append(player_averages.get(player['name'], 0))
            actual_kills.append(player['kills'])
    return predictions, actual_kills

def evaluate_predictions(all_predictions, all_actual):
    mse = mean_squared_error(all_actual, all_predictions)
    mae = mean_absolute_error(all_actual, all_predictions)
    r2 = r2_score(all_actual, all_predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

def main():
    # Get all maps
    all_maps = get_maps()
    
    # Split maps into train and test sets
    train_maps, test_maps = split_maps(all_maps)
    
    # Calculate player averages from train set
    player_averages = calculate_player_averages(train_maps)
    
    # Make predictions on test set
    all_predictions = []
    all_actual = []
    for map_data in test_maps:
        predictions, actual = predict_kills(player_averages, map_data)
        all_predictions.extend(predictions)
        all_actual.extend(actual)
    
    # Evaluate predictions
    print("Evaluation on Test Set:")
    evaluate_predictions(all_predictions, all_actual)
    

if __name__ == "__main__":
    main()