import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from match_dataset import MatchDataset  # Make sure to import your dataset class
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


"""This is a model that is meant to model a CSGO match. 
The input is just a one hot encoding of the players on each team, the out put will be the number of kills each player on each team gets, 
the purpose is to model how specific unique players perform against eachother. """
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        self.player_embedding = nn.Embedding(num_players, 32)
        
        self.network = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_players)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        
    def forward(self, x):
        batch_size, _, num_players = x.size()
        
        player_indices = torch.arange(num_players).unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1).to(x.device)
        
        player_emb = self.player_embedding(player_indices)
        
        player_emb = player_emb * x.unsqueeze(-1)
        
        player_emb = player_emb.sum(dim=2)
        
        combined = player_emb.view(batch_size, -1)
        
        output = self.network(combined)
        
        return output

def custom_loss(pred, target, active_mask):
    # Huber loss for active players only
    delta = 1.0
    loss = torch.where(active_mask,
                       torch.where(torch.abs(target - pred) < delta,
                                   0.5 * (target - pred)**2,
                                   delta * (torch.abs(target - pred) - 0.5 * delta)),
                       torch.zeros_like(pred))
    return loss.sum() / (active_mask.sum() + 1e-8)  # Add small epsilon to avoid division by zero

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            active_mask = (batch_features.sum(dim=1) != 0)
            loss = criterion(outputs, batch_labels, active_mask)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                active_mask = (batch_features.sum(dim=1) != 0)
                loss = criterion(outputs, batch_labels, active_mask)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print("Training completed.")

def evaluate_model(model, test_loader, criterion, device, dataset):
    model.eval()
    test_loss = 0.0
    all_active_predictions = []
    all_active_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            active_mask = (batch_features.sum(dim=1) != 0)
            loss = criterion(outputs, batch_labels, active_mask)
            test_loss += loss.item()
            
            active_predictions = outputs[active_mask].detach().cpu().numpy()
            active_labels = batch_labels[active_mask].cpu().numpy()
            
            all_active_predictions.extend(active_predictions)
            all_active_labels.extend(active_labels)
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    all_active_predictions = np.array(all_active_predictions)
    all_active_labels = np.array(all_active_labels)
    
    # Calculate metrics for active players only
    mse = nn.MSELoss()(torch.tensor(all_active_predictions), torch.tensor(all_active_labels))
    mae = mean_absolute_error(all_active_labels, all_active_predictions)
    r2 = r2_score(all_active_labels, all_active_predictions)
    
    print(f"Mean Squared Error (Active Players): {mse:.4f}")
    print(f"Mean Absolute Error (Active Players): {mae:.4f}")
    print(f"R-squared Score (Active Players): {r2:.4f}")


def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MatchDataset(threshold=15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the improved model
    model = ImprovedNeuralNetwork(num_players=dataset.num_players).to(device)

    # Custom loss function and optimizer
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, criterion, device, dataset)

if __name__ == "__main__":
    main()