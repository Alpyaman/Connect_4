import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from src.model import Connect4Net
import os

# --- CONFIG ---
BATCH_SIZE = 64
LEARNING_RATE = 0.0005 # Lower LR for fine-tuning
EPOCHS = 10
DATA_FILE = "self_play_data.pkl"
MODEL_FILE = "connect4_cnn.pth" # We load this, update it, and save it back

def augment_data(boards, winners):
    """
    Creates a balanced dataset by swapping P1 and P2 tokens.
    If Board X -> Winner 1
    Then Swapped(Board X) -> Winner -1
    """
    print(f"Augmenting data... Initial size: {len(boards)}")
    
    aug_boards = []
    aug_winners = []
    
    for board, winner in zip(boards, winners):
        # Add Original
        aug_boards.append(board)
        aug_winners.append(winner)
        
        # Create Swapped (Invert Perspective)
        # Original: 1 (P1), -1 (P2), 0 (Empty)
        # Swap: 1 -> -1, -1 -> 1, 0 -> 0
        swapped_board = board * -1
        flipped_winner = winner * -1
        
        aug_boards.append(swapped_board)
        aug_winners.append(flipped_winner)
        
    print(f"Augmentation complete. New size: {len(aug_boards)}")
    return np.array(aug_boards), np.array(aug_winners)

def load_data():
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    # Extract
    boards_flat = np.array([d['board'] for d in data], dtype=np.float32)
    winners = np.array([d['value'] for d in data], dtype=np.float32)

    # Reshape
    boards = boards_flat.reshape(-1, 1, 6, 7)

    # Fix Labels: Ensure P2 is -1 (just in case)
    # Our self_play script saved P1=1, P2=-1, so this is likely already correct.
    # But let's be safe.
    
    # Run Augmentation
    boards, winners = augment_data(boards, winners)

    X_tensor = torch.tensor(boards)
    y_tensor = torch.tensor(winners).view(-1, 1)

    return X_tensor, y_tensor

def retrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Retraining on: {device}")

    X, y = load_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Existing Model
    model = Connect4Net().to(device)
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print(f"Loaded existing weights from {MODEL_FILE}")
    else:
        print("Warning: No existing model found. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\nStarting Fine-Tuning...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save Update
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\nModel updated and saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    retrain()