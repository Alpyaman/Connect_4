import numpy as np
import torch
import pickle
from src.model import Connect4Net
import src.mcts as mcts
import os
from tqdm import tqdm

# --- CONFIG ---
GAMES_TO_PLAY = 100  # Start small to test. Increase to 1000+ for real training.
MCTS_SIMS = 50       # Lower sims for faster data generation (50-100 is okay for "Gen 1")
OUTPUT_FILE = "self_play_data.pkl"
MODEL_PATH = "connect4_cnn.pth"

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r

def is_win(board, piece):
    # Reuse the logic from mcts or play.py
    # Horizontal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Pos Diag
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Neg Diag
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def self_play(model, device):
    board = create_board()
    game_history = [] # Stores (board_state, player_turn)
    turn = 1 # 1 or -1
    game_over = False
    
    while not game_over:
        # Save current state for training later
        # We save the board relative to the player whose turn it is
        # But for now, let's just save raw board and figure out winner later
        game_history.append((board.copy(), turn))
        
        # Temporary hack: override MCTS sims for speed
        original_sims = mcts.SIMULATIONS
        mcts.SIMULATIONS = MCTS_SIMS
        
        # AI makes a move using MCTS
        col = mcts.mcts_search(board, model, device)
        
        # Restore sims (good practice)
        mcts.SIMULATIONS = original_sims
        
        row = get_next_open_row(board, col)
        board[row][col] = turn

        if is_win(board, turn):
            return game_history, turn # Return data and the winner (1 or -1)
        
        # Check Draw
        if 0 not in board[0]:
            return game_history, 0 # Draw
            
        turn *= -1

def generate_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating data on {device}...")

    # Load the current best brain
    model = Connect4Net().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("No model found! Using random weights (untrained).")

    all_boards = []
    all_values = []
    
    # Statistics
    p1_wins = 0
    p2_wins = 0
    draws = 0

    print(f"Starting self-play for {GAMES_TO_PLAY} games...")
    for _ in tqdm(range(GAMES_TO_PLAY)):
        history, winner = self_play(model, device)
        
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1

        # Process the game history into training samples
        for state, player_turn in history:
            all_boards.append(state)
            
            # If P1 won (winner=1):
            #   - States where it was P1's turn are GOOD (Target = 1)
            #   - States where it was P2's turn are BAD (Target = -1)
            # Logic: Value = Winner * Player_Turn
            # Example: Winner=1, Turn=1 -> Value=1 (Good for P1)
            # Example: Winner=1, Turn=-1 -> Value=-1 (Bad for P2)
            # Example: Winner=-1, Turn=1 -> Value=-1 (Bad for P1)
            
            if winner == 0:
                value = 0.0
            else:
                value = float(winner) # We want the board to predict the FINAL WINNER
            
            all_values.append(value)

    # Save to file
    data = []
    for b, v in zip(all_boards, all_values):
        data.append({'board': b.flatten(), 'value': v})
        
    print(f"Saving {len(data)} positions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data, f)
        
    print("--- Stats ---")
    print(f"P1 Wins: {p1_wins}")
    print(f"P2 Wins: {p2_wins}")
    print(f"Draws:   {draws}")

if __name__ == "__main__":
    generate_dataset()