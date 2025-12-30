import torch
import numpy as np
import os
import time

from src.model import Connect4Net
import src.mcts as mcts  # Import our new Brain

# --- CONFIG ---
MODEL_PATH = "connect4_cnn.pth"
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER = 1  # You
AI = -1     # The Model

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r

def print_board(board):
    print("\n  0   1   2   3   4   5   6 ")
    print("-----------------------------")
    symbols = {0: ' . ', 1: ' X ', -1: ' O '}
    for r in range(ROW_COUNT):
        row_str = "|"
        for c in range(COLUMN_COUNT):
            row_str += symbols[board[r][c]] + "|"
        print(row_str)
    print("-----------------------------")

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
    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def main():
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Brain loaded successfully!")
    else:
        print("Model file not found. Run train.py first.")
        return

    board = create_board()
    game_over = False
    turn = 0 # 0 = Player, 1 = AI

    print_board(board)

    while not game_over:
        if turn == 0:
            # Player Turn
            try:
                col = int(input("Player 1 (X) Choose column (0-6): "))
                if col < 0 or col > 6:
                    print("Invalid column.")
                    continue
            except ValueError:
                print("Please enter a number.")
                continue

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER)

                if winning_move(board, PLAYER):
                    print_board(board)
                    print("PLAYER 1 WINS!!")
                    game_over = True
                
                turn = 1
                print_board(board)
            else:
                print("Column full! Try another.")

        else:
            # AI Turn (MCTS)
            print("AI is Thinking (MCTS)...")
            start_time = time.time()
            
            # CALL MCTS HERE
            col = mcts.mcts_search(board, model, device)
            
            end_time = time.time()
            print(f"AI chose column {col} in {end_time - start_time:.2f} seconds.")

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI)

                if winning_move(board, AI):
                    print_board(board)
                    print("AI WINS!! Better luck next time.")
                    game_over = True

                turn = 0
                print_board(board)

if __name__ == "__main__":
    main()