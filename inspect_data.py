import pickle
import numpy as np
import random

FILENAME = "connect4_dataset.pkl"

def print_board(board_flat):
    # RESHAPE: Turn flat array (42,) into grid (6, 7)
    board = board_flat.reshape(6, 7)

    # MAP: 0=Empty, 1=P1, 2=P2 (and -1 just in case)
    symbols = {0: ' . ', 1: ' X ', 2: ' O ', -1: ' O '}
    
    print("---------------------")
    for row in board:
        # visualizer loop
        line = []
        for x in row:
            # We cast to standard int to avoid numpy type issues
            val = int(x) 
            line.append(symbols.get(val, ' ? '))
        print("".join(line))
    print("---------------------")

def inspect():
    print(f"Loading {FILENAME}...")
    with open(FILENAME, 'rb') as f:
        data = pickle.load(f)

    # Extract data (List of Dicts format)
    boards = np.array([d['board'] for d in data])
    winners = np.array([d['value'] for d in data])

    print("\n--- DATA SHAPES ---")
    print(f"Boards (Flat): {boards.shape}") 
    
    # Check stats
    unique, counts = np.unique(winners, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"P1: {dist.get(1,0)} | P2: {dist.get(-1,0)} | Draw: {dist.get(0,0)}")

    # Visual Check
    print("\n--- RANDOM SAMPLE ---")
    idx = random.randint(0, len(boards) - 1)
    print(f"Label: {winners[idx]}")
    print_board(boards[idx])

if __name__ == "__main__":
    inspect()