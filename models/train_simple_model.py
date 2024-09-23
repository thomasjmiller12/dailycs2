import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from match_dataset import MatchDataset
from simple_model import NeuralNetwork
from training_utils import custom_loss, train_model, evaluate_model, setup_data

def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    dataset, train_loader, val_loader, test_loader = setup_data(MatchDataset, threshold=10, batch_size=batch_size)

    # Initialize the model
    model = NeuralNetwork(num_players=dataset.num_players).to(device)

    # Custom loss function and optimizer
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 'simple')

    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, criterion, device, dataset, 'simple')

if __name__ == "__main__":
    main()