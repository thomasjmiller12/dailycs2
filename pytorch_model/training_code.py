import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from match_dataset import MatchDataset  # Make sure to import your dataset class
from simple_model import NeuralNetwork  # Make sure to import your model class
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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
            
            # Filter predictions and labels for active players
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
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split the dataset
    dataset = MatchDataset(threshold=10)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    model = NeuralNetwork(num_players=dataset.num_players).to(device)

    # Custom loss function and optimizer
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, criterion, device, dataset)

if __name__ == "__main__":
    main()