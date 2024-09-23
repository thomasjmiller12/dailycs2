import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from match_dataset import DenseMatchDataset
from emb_model import EmbeddingModel
from training_utils import train_model, evaluate_model, setup_data
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import argparse

def evaluate_top_n_players(model, test_loader, criterion, device, dataset, top_n):
    model.eval()
    test_loss = 0.0
    all_active_predictions = []
    all_active_labels = []
    
    top_n_players = dataset.get_top_n_players(top_n)
    top_n_indices = [dataset.player_to_index[player] for player in top_n_players]
    
    print(f"Top {top_n} players: {top_n_players}")
    print(f"Top {top_n} indices: {top_n_indices}")
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:

            top_n_mask = torch.zeros_like(batch_features, dtype=torch.bool)
            for i in top_n_indices:
                top_n_mask |= (batch_features == i)
            if not top_n_mask.any():
                continue
            print(top_n_mask.sum())
            
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            outputs = outputs.detach().cpu()
            
            active_predictions = torch.masked_select(outputs, top_n_mask)
            active_labels = torch.masked_select(batch_labels, top_n_mask)
            
            all_active_predictions.extend(active_predictions.numpy())
            all_active_labels.extend(active_labels.numpy())
    
    print(len(all_active_predictions))
    all_active_predictions = np.array(all_active_predictions)
    all_active_labels = np.array(all_active_labels)
    
    # Calculate metrics for top N players only
    mse = nn.MSELoss()(torch.tensor(all_active_predictions), torch.tensor(all_active_labels))
    mae = mean_absolute_error(all_active_labels, all_active_predictions)
    r2 = r2_score(all_active_labels, all_active_predictions)
    
    print(f"Mean Squared Error (Top {top_n} Players): {mse:.4f}")
    print(f"Mean Absolute Error (Top {top_n} Players): {mae:.4f}")
    print(f"R-squared Score (Top {top_n} Players): {r2:.4f}")

def main(top_n=None):
    # Hyperparameters
    batch_size = 512
    learning_rate = 0.001
    num_epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    dataset, train_loader, val_loader, test_loader = setup_data(DenseMatchDataset, threshold=25, batch_size=batch_size)
    print(f"Number of players: {dataset.num_players}")
    # Initialize the model
    model = EmbeddingModel(num_players=dataset.num_players, batch_size=batch_size).to(device)

    # Standard MSE loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 'embedding')

    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, criterion, device, dataset, 'embedding')
    
    # Optionally evaluate top N players
    if top_n is not None:
        evaluate_top_n_players(model, test_loader, criterion, device, dataset, top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the embedding model.')
    parser.add_argument('--top_n', type=int, default=100, help='Evaluate top N players based on data points')
    args = parser.parse_args()
    main(top_n=args.top_n)