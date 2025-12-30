import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from src.model import Connect4Net

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20  # You can increase this later
FILENAME = "connect4_dataset.pkl"

def load_data():
    print(f"Loading {FILENAME}...")
    with open(FILENAME, 'rb') as f:
        data = pickle.load(f)

    # 1. Extract and convert to Numpy
    # Depending on your generate_data.py, data is a list of dicts
    boards_flat = np.array([d['board'] for d in data], dtype=np.float32)
    winners = np.array([d['value'] for d in data], dtype=np.float32)

    # 2. Reshape: (N, 42) -> (N, 1, 6, 7) for CNN
    # N images, 1 Channel, 6 Rows, 7 Cols
    boards = boards_flat.reshape(-1, 1, 6, 7)

    # 3. Normalize: Convert 2 (Player 2) to -1
    # Current: 0, 1, 2
    # Target:  0, 1, -1
    boards[boards == 2] = -1

    # 4. Convert to PyTorch Tensors
    X_tensor = torch.tensor(boards)
    y_tensor = torch.tensor(winners).view(-1, 1) # Shape (N, 1)

    print(f"Data ready. Input shape: {X_tensor.shape}")
    return X_tensor, y_tensor

def train():
    # Setup Device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data
    X, y = load_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = Connect4Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error (since we predict a score -1 to 1)

    # Training Loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "connect4_cnn.pth")
    print("\nModel saved to 'connect4_cnn.pth'")

if __name__ == "__main__":
    train()