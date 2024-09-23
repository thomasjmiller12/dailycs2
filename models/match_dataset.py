import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db import get_maps

def get_unique_players(maps_data, threshold=5):
    players = set(['generic'])
    player_datapoint_count = Counter()
    
    for map_data in maps_data:
        for team in map_data['teams']:
            for player in team['players']:
                if player['name']:
                    player_datapoint_count[player['name']] += 1
                    if player_datapoint_count[player['name']] >= threshold:
                        players.add(player['name'])
    
    player_to_index = {player: idx for idx, player in enumerate(players)}
    return players, player_to_index, player_datapoint_count

class MatchDataset(Dataset):
    def __init__(self, threshold=5):
        # Use get_maps function to retrieve data
        maps_data = get_maps()
        
        self.players, self.player_to_index, self.player_datapoint_count = get_unique_players(maps_data, threshold)
        self.processed_data = []

        for map_data in maps_data:
            skipmap = False
            label = torch.zeros(len(self.players))
            feature_vector = torch.zeros((2, len(self.players)))
            for i, team in enumerate(map_data['teams']):
                if len(team['players']) > 5:
                    skipmap = True
                    break
                for player in team['players']:
                    if player['kills'] == 0:
                        skipmap = True
                    if player['name'] in self.players:
                        feature_vector[i][self.player_to_index[player['name']]] = 1
                        label[self.player_to_index[player['name']]] = player['kills']
                    else: 
                        feature_vector[i][self.player_to_index['generic']] = 1
                        label[self.player_to_index['generic']] = player['kills']
                        self.player_datapoint_count['generic'] += 1
            if not skipmap:
                self.processed_data.append((feature_vector, label))

    def __getitem__(self, idx):
        # x,y where x is a (2,num_players tensor), where 2 is just the dimension of teams and the players are one hot encoded per team. Then Y is (num_players,) which is the number of kills that player got during the match
        return self.processed_data[idx]

    def __len__(self):
        return len(self.processed_data)
    
    @property
    def num_players(self):
        return len(self.players)
    
    def get_player_datapoint_counts(self):
        return dict(self.player_datapoint_count)
    
    def get_top_n_players(self, n):
        # Exclude 'generic' player
        filtered_counts = {player: count for player, count in self.player_datapoint_count.items() if player != 'generic'}
        return [player for player, _ in Counter(filtered_counts).most_common(n)]
    
class DenseMatchDataset(Dataset):
    def __init__(self, threshold=5):
        # Use get_maps function to retrieve data
        maps_data = get_maps()
        
        self.players, self.player_to_index, self.player_datapoint_count = get_unique_players(maps_data, threshold)
        self.processed_data = []
        
        for map_data in maps_data:
            skipmap = False
            feature_vector = torch.zeros((2, 5), dtype=torch.int64)
            label = torch.zeros((2, 5), dtype=torch.float)
            
            for i, team in enumerate(map_data['teams']):
                if len(team['players']) > 5:
                    skipmap = True
                    break
                for j, player in enumerate(team['players']):
                    if player['kills'] == 0:
                        skipmap = True
                    if player['name'] in self.players:
                        feature_vector[i][j] = self.player_to_index[player['name']]
                    else:
                        feature_vector[i][j] = self.player_to_index['generic']
                        self.player_datapoint_count['generic'] += 1
                    label[i][j] = player['kills']
            
            if not skipmap:
                self.processed_data.append((feature_vector, label))

    def __getitem__(self, idx):
        # x,y where x is a (2,5) tensor of player indices, and y is a (2,5) tensor of kill counts
        return self.processed_data[idx]

    def __len__(self):
        return len(self.processed_data)
    
    @property
    def num_players(self):
        return len(self.players)
    
    def get_player_datapoint_counts(self):
        return dict(self.player_datapoint_count)
    
    def get_top_n_players(self, n):
        # Exclude 'generic' player
        filtered_counts = {player: count for player, count in self.player_datapoint_count.items() if player != 'generic'}
        return [player for player, _ in Counter(filtered_counts).most_common(n)]

def main():
    # Load MatchDataset
    match_dataset = MatchDataset(threshold=5)
    print(f"MatchDataset size: {len(match_dataset)}")
    print(f"MatchDataset example:")
    example_feature, example_label = match_dataset[0]
    print(f"  Feature: {example_feature}")
    print(f"  Label: {example_label}")
    print(f"MatchDataset input shape: {example_feature.shape}")
    print(f"MatchDataset output shape: {example_label.shape}")
    print()

    # Load DenseMatchDataset
    dense_dataset = DenseMatchDataset(threshold=5)
    print(f"DenseMatchDataset size: {len(dense_dataset)}")
    print(f"DenseMatchDataset example:")
    example_feature, example_label = dense_dataset[0]
    print(f"  Feature: {example_feature}")
    print(f"  Label: {example_label}")
    print(f"DenseMatchDataset input shape: {example_feature.shape}")
    print(f"DenseMatchDataset output shape: {example_label.shape}")

if __name__ == "__main__":
    main()