import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from match_dataset import FocusPlayerDataset
from emb_model import EmbeddingModel
from training_utils import evaluate_model, setup_data
import numpy as np
import argparse

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for (teammates, opponents, focus_player_idx), labels in train_loader:
            # Move data to device
            teammates = teammates.to(device)
            opponents = opponents.to(device)
            focus_player_idx = focus_player_idx.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model((teammates, opponents, focus_player_idx))
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (teammates, opponents, focus_player_idx), labels in val_loader:
                teammates = teammates.to(device)
                opponents = opponents.to(device)
                focus_player_idx = focus_player_idx.to(device)
                labels = labels.to(device)
                
                outputs = model((teammates, opponents, focus_player_idx))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    dataset, train_loader, val_loader, test_loader = setup_data(
        FocusPlayerDataset, 
        threshold=15, 
        batch_size=batch_size
    )
    
    # Initialize model
    model = EmbeddingModel(num_players=dataset.num_players, embedding_dim=8).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

if __name__ == "__main__":
    main()
