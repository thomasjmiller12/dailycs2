import os
import torch
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_players, batch_size):
        super().__init__()
        self.embedding = nn.Embedding(num_players, 16)
        self.batch_size = batch_size
        self.attention_dim = 16
        
        # Self-attention for own team
        self.self_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True, dropout=0.05)
        # Cross-attention for opposite team
        self.cross_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True, dropout=0.05)
        
        # Linear layers for Q, K, V projections
        self.query_proj = nn.Linear(16, 16)
        self.key_proj = nn.Linear(16, 16)
        self.value_proj = nn.Linear(16, 16)
        
        # Final linear layer for kill prediction
        self.final_linear = nn.Linear(16, 1)

        # Print total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def forward(self, x):
        # x should be a tensor of shape (batch_size, 2, 5)
        batch_size, num_teams, players_per_team = x.shape

        # embeddings is a tensor of shape (batch_size, 2, 5, 16)
        embeddings = self.embedding(x)

        # Self-attention within each team
        self_attn_output = []
        for i in range(num_teams):
            team_embeddings = embeddings[:, i]  # Shape: (batch_size, 5, 16)
            Q = self.query_proj(team_embeddings)
            K = self.key_proj(team_embeddings)
            V = self.value_proj(team_embeddings)
            attn_output, _ = self.self_attention(Q, K, V)
            self_attn_output.append(attn_output)
        
        self_attn_output = torch.stack(self_attn_output, dim=1)  # Shape: (batch_size, 2, 5, 16)

        # Cross-attention between teams
        cross_attn_output = []
        for i in range(num_teams):
            query = self.query_proj(self_attn_output[:, i])
            key = self.key_proj(self_attn_output[:, 1-i])  # 1-i gives the opposite team index
            value = self.value_proj(self_attn_output[:, 1-i])
            attn_output, _ = self.cross_attention(query, key, value)
            cross_attn_output.append(attn_output)

        cross_attn_output = torch.stack(cross_attn_output, dim=1)

        # Combine self-attention and cross-attention outputs
        combined_output = self_attn_output + cross_attn_output

        # Final linear layer for kill prediction
        kills_prediction = self.final_linear(combined_output).squeeze(-1)

        return kills_prediction  # Shape: (batch_size, 2, 5)


def main():
    from match_dataset import DenseMatchDataset
    from torch.utils.data import DataLoader

    # Load the DenseMatchDataset
    dataset = DenseMatchDataset(threshold=5)
    
    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = EmbeddingModel(num_players=dataset.num_players, batch_size=batch_size)

    # Get a batch of data
    features, labels = next(iter(dataloader))

    print(f"Input shape: {features.shape}")
    
    # Forward pass
    output = model(features)
    print(f"Output shape: {output.shape}")

    # Print embedding for first player in first team of first batch
    print(f"Embedding for player 0: {output[0, 0, 0]}")

if __name__ == "__main__":
    main()