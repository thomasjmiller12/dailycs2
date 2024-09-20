import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter
from db.db import get_maps

class MatchDataset(Dataset):
    def __init__(self, threshold=5):
        self.players = set()
        self.player_to_index = {}
        self.processed_data = []
        self.player_datapoint_count = Counter()
        
        self.players.add('generic')
        
        # Use get_maps function to retrieve data
        maps_data = get_maps()
        
        for map_data in maps_data:
            for team in map_data['teams']:
                for player in team['players']:
                    if player['name']:
                        self.player_datapoint_count[player['name']] += 1
                        if self.player_datapoint_count[player['name']] >= threshold:
                            self.players.add(player['name'])
        
        self.player_to_index = {player: idx for idx, player in enumerate(self.players)}

        for map_data in maps_data:
            skipmap = False
            label = torch.zeros(len(self.players))
            feature_vector = torch.zeros((2, len(self.players)))
            for i, team in enumerate(map_data['teams']):
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
    

def main():
    dataset = MatchDataset(threshold=10)

    # Create a DataLoader
    batch_size = 64
    shuffle = True
    num_workers = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Example of iterating through the DataLoader
    for batch_features, batch_labels in dataloader:
        print(f"Batch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        # Here you would typically pass this batch to your model for training or inference
        break  # This break is just to show the first batch; remove it for actual training

    # Additional information
    print(f"Number of unique players: {dataset.num_players}")
    print(f"Number of batches: {len(dataloader)}")

if __name__ == "__main__":
    main()