import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.model import Connect4Net
import random

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints"
GAMES_PER_GEN = 50  # Games to play for benchmarking each version

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r

def winning_move(board, piece):
    # Check horizontal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Check diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def get_raw_network_move(board, model, device):
    # Pure "Intuition" move (No MCTS) - very fast
    valid_cols = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
    best_score = -float('inf')
    best_col = random.choice(valid_cols)
    
    # AI is Player -1. It wants the output to be -1.
    # But wait! In our data, P2 wins are -1. 
    # So if the model outputs -0.9, that means "I (P2) am winning".
    # So we want to MINIMIZE the output score.
    
    # Actually, simpler: Let's convert board to tensor and check "Next State Value"
    for col in valid_cols:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        temp_board[row][col] = -1 # AI Move
        
        # Prepare input
        tensor = torch.tensor(temp_board, dtype=torch.float32).reshape(1, 1, 6, 7).to(device)
        with torch.no_grad():
            score = model(tensor).item()
            
        # We want the score to be as close to -1 (AI Win) as possible
        # So we want to MINIMIZE score
        if score < 0 and score < best_score: 
             # Wait, logic check: -0.9 is better than -0.1. 
             # If we initialize best_score to +inf, we minimize.
             pass
    
    # Let's use simple logic:
    # We want the move that results in the LOWEST value (closest to -1)
    best_val = float('inf')
    for col in valid_cols:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        temp_board[row][col] = -1 
        
        tensor = torch.tensor(temp_board, dtype=torch.float32).reshape(1, 1, 6, 7).to(device)
        with torch.no_grad():
            val = model(tensor).item()
        
        if val < best_val:
            best_val = val
            best_col = col
            
    return best_col

def play_game(model, device):
    board = create_board()
    game_over = False
    turn = random.randint(0, 1) # Random start
    
    while not game_over:
        if turn == 0:
            # Random Player (Player 1)
            valid = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
            col = random.choice(valid)
            row = get_next_open_row(board, col)
            board[row][col] = 1
            if winning_move(board, 1):
                return 1 # P1 wins
            
        else:
            # AI Player (Player -1)
            col = get_raw_network_move(board, model, device)
            row = get_next_open_row(board, col)
            board[row][col] = -1
            if winning_move(board, -1):
                return -1 # AI wins
            
        if 0 not in board[0]:
            return 0 # Draw
        turn = 1 - turn

def benchmark():
    if not os.path.exists(CHECKPOINT_DIR):
        print("No checkpoints found. Run auto_train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
    
    # Sort files naturally (gen_1, gen_2, ... gen_10)
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    gens = []
    win_rates = []
    
    print(f"{'Gen':<5} | {'Wins':<5} | {'Loss':<5} | {'Draw':<5} | {'Win Rate'}")
    print("-" * 45)

    for f in files:
        gen_num = int(f.split('_')[-1].split('.')[0])
        model_path = os.path.join(CHECKPOINT_DIR, f)
        
        model = Connect4Net().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        ai_wins = 0
        losses = 0
        draws = 0
        
        for _ in range(GAMES_PER_GEN):
            result = play_game(model, device)
            if result == -1:
                ai_wins += 1
            elif result == 1:
                losses += 1
            else:
                draws += 1
            
        rate = (ai_wins / GAMES_PER_GEN) * 100
        print(f"{gen_num:<5} | {ai_wins:<5} | {losses:<5} | {draws:<5} | {rate:.1f}%")
        
        gens.append(gen_num)
        win_rates.append(rate)

    # Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(gens, win_rates, marker='o', linestyle='-', color='b')
        plt.title("AI Improvement: Win Rate vs Random Agent")
        plt.xlabel("Generation")
        plt.ylabel("Win Rate (%)")
        plt.grid(True)
        plt.savefig("progress_graph.png")
        print("\nGraph saved to 'progress_graph.png'")
        plt.show()
    except Exception as e:
        print(f"\nCould not generate plot (matplotlib missing?), but stats are above. {e}")

if __name__ == "__main__":
    benchmark()