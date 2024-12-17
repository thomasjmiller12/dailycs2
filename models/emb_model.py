import os
import torch
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_players, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_players, embedding_dim)
        
        # Cross-attention between focus player and teammates
        self.teammate_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=2,
            batch_first=True,
            dropout=0.2  # Increased dropout
        )
        
        # Cross-attention between focus player and opponents  
        self.opponent_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4, 
            batch_first=True,
            dropout=0.2  # Increased dropout
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Combine all information
        combined_dim = embedding_dim * 3  # focus_player + teammate_context + opponent_context
        
        # Simplified prediction layers with more regularization
        self.prediction_layers = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),  # Added layer norm
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(embedding_dim, 32),  # Added wider layer
            nn.LayerNorm(32),  # Added layer norm
            nn.ReLU(), 
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(32, 1)
        )

        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Print total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def forward(self, data):
        teammates, opponents, focus_player_idx = data
        batch_size = teammates.shape[0]
        
        # Get embeddings
        focus_player_emb = self.embedding(focus_player_idx)  # (batch, embedding_dim)
        teammate_emb = self.embedding(teammates).squeeze(1)  # (batch, 4, embedding_dim)
        opponent_emb = self.embedding(opponents).squeeze(1)  # (batch, 5, embedding_dim)
        
        # Expand focus player embedding for attention
        focus_player_expanded = focus_player_emb.unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # Cross-attention between focus player and teammates
        teammate_context, _ = self.teammate_attention(
            focus_player_expanded,  # query
            teammate_emb,           # key
            teammate_emb            # value
        )
        teammate_context = self.layer_norm1(teammate_context)
        teammate_context = teammate_context.squeeze(1)  # (batch, embedding_dim)
        
        # Cross-attention between focus player and opponents
        opponent_context, _ = self.opponent_attention(
            focus_player_expanded,  # query
            opponent_emb,           # key
            opponent_emb            # value
        )
        opponent_context = self.layer_norm2(opponent_context)
        opponent_context = opponent_context.squeeze(1)  # (batch, embedding_dim)
        
        # Combine all contexts
        combined = torch.cat([
            focus_player_emb,      # Original focus player embedding
            teammate_context,       # Context from teammate attention
            opponent_context        # Context from opponent attention
        ], dim=1)
        
        # Predict kills for focus player
        prediction = self.prediction_layers(combined).squeeze(-1)
        
        return prediction


def main():
    from match_dataset import FocusPlayerDataset
    from torch.utils.data import DataLoader

    # Load the FocusPlayerDataset
    dataset = FocusPlayerDataset(threshold=5)
    
    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = EmbeddingModel(num_players=dataset.num_players)

    # Get a batch of data
    (teammates, opponents, focus_player_idx), labels = next(iter(dataloader))

    print(f"Teammates shape: {teammates.shape}")
    print(f"Opponents shape: {opponents.shape}")
    print(f"Focus player shape: {focus_player_idx.shape}")
    
    # Forward pass
    output = model((teammates, opponents, focus_player_idx))
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
