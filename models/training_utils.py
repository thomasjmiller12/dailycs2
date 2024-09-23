import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_type):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            if model_type == 'simple':
                active_mask = (batch_features.sum(dim=1) != 0)
                loss = criterion(outputs, batch_labels, active_mask)
            else:  # embedding model
                loss = criterion(outputs, batch_labels)
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
                if model_type == 'simple':
                    active_mask = (batch_features.sum(dim=1) != 0)
                    loss = criterion(outputs, batch_labels, active_mask)
                else:  # embedding model
                    loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print("Training completed.")
    
    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

def evaluate_model(model, test_loader, criterion, device, dataset, model_type):
    model.eval()
    test_loss = 0.0
    all_active_predictions = []
    all_active_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            if model_type == 'simple':
                active_mask = (batch_features.sum(dim=1) != 0)
                loss = criterion(outputs, batch_labels, active_mask)
            else:  # embedding model
                loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            
            # Filter predictions and labels for active players
            if model_type == 'simple':
                active_predictions = outputs[active_mask].detach().cpu().numpy()
                active_labels = batch_labels[active_mask].cpu().numpy()
            else:  # embedding model
                active_predictions = outputs.detach().cpu().numpy().flatten()
                active_labels = batch_labels.cpu().numpy().flatten()
            
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

def setup_data(dataset_class, threshold, batch_size):
    # Load the dataset
    dataset = dataset_class(threshold=threshold)
    
    # Split the dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, train_loader, val_loader, test_loader