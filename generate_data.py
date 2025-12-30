import random
import pickle
import time
from tqdm import tqdm

# --- CONFIGURATION ---
NUM_GAMES = 1000      # Start with 1000 to test speed, then scale up
epsilon = 0.1         # 10% chance to make a random move (adds variety to dataset)
OUTPUT_FILE = "connect4_dataset.pkl"

# --- GAME CONSTANTS ---
ROWS = 6
COLS = 7

# --- BITBOARD LOGIC (Reusing your optimized engine) ---
def make_bitboard(board_list, player_mark):
    position = 0
    mask = 0
    for c in range(COLS):
        for r in range(ROWS):
            idx = r * COLS + c
            if board_list[idx] != 0:
                mask |= (1 << (c * (ROWS + 1) + (ROWS - 1 - r)))
                if board_list[idx] == player_mark:
                    position |= (1 << (c * (ROWS + 1) + (ROWS - 1 - r)))
    return position, mask

def get_valid_moves(mask):
    moves = []
    # Standard column order
    for col in range(COLS):
        if (mask & (1 << (col * (ROWS + 1) + ROWS - 1))) == 0:
            moves.append(col)
    return moves

def connected_four(position):
    # Horizontal
    m = position & (position >> (ROWS + 1))
    if m & (m >> (2 * (ROWS + 1))):
        return True
    # Diagonal \
    m = position & (position >> ROWS)
    if m & (m >> (2 * ROWS)):
        return True
    # Diagonal /
    m = position & (position >> (ROWS + 2))
    if m & (m >> (2 * (ROWS + 2))):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    return False

# --- THE MINIMAX AGENT (Condensed for this script) ---
# We use a simpler version here without strict time checks for data gen speed
# but kept the alpha-beta logic.

TRANS_TABLE = {}

def negamax(position, mask, depth, alpha, beta):
    key = (position, mask)
    if key in TRANS_TABLE and TRANS_TABLE[key][1] >= depth:
        return TRANS_TABLE[key][0]

    if connected_four(position ^ mask): # Opponent won
        return -(10000 + depth)

    if depth == 0 or mask == ((1 << ((ROWS + 1) * COLS)) - 1) ^ ((1 << ((ROWS + 1) * COLS)) // ((1 << (ROWS + 1)) - 1)):
        return 0

    valid_moves = get_valid_moves(mask)
    # Heuristic ordering: Center first
    valid_moves.sort(key=lambda x: abs(x - 3))
    
    best_score = -float('inf')

    for col in valid_moves:
        new_pos = position ^ mask
        new_mask = mask | (mask + (1 << (col * (ROWS + 1))))
        score = -negamax(new_pos, new_mask, depth - 1, -beta, -alpha)
        
        if score > best_score:
            best_score = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    TRANS_TABLE[key] = (best_score, depth)
    return best_score

def get_best_move(board_list, mark, depth=4):
    """
    Returns the best move using Minimax.
    depth: Search depth. Lower = faster generation, Higher = better quality data.
    """
    position, mask = make_bitboard(board_list, mark)
    valid_moves = get_valid_moves(mask)
    
    # Epsilon-Greedy: Sometimes play random for dataset diversity
    if random.random() < epsilon:
        return random.choice(valid_moves)

    best_move = valid_moves[0]
    max_score = -float('inf')
    
    # Randomize order of equal moves to avoid repetitive games
    random.shuffle(valid_moves)
    valid_moves.sort(key=lambda x: abs(x - 3)) # Re-sort by center preference

    for col in valid_moves:
        new_pos = position ^ mask
        new_mask = mask | (mask + (1 << (col * (ROWS + 1))))
        score = -negamax(new_pos, new_mask, depth-1, -float('inf'), float('inf'))
        
        if score > max_score:
            max_score = score
            best_move = col
            
    return best_move

# --- DATA GENERATION LOOP ---

def play_game():
    # Board: 0=Empty, 1=P1, 2=P2
    board = [0] * (ROWS * COLS)
    history = [] # Stores (board_state, current_player)
    
    player = 1
    
    while True:
        # 1. Record state before move
        # Copy board to avoid reference issues
        history.append((list(board), player))
        
        # 2. Get Move
        # Note: We use depth=2 or 3 for speed. depth=4 is better but slower.
        col = get_best_move(board, player, depth=2)
        
        # 3. Apply Move
        # Find row
        row_idx = -1
        for r in range(ROWS-1, -1, -1):
            idx = r * COLS + col
            if board[idx] == 0:
                row_idx = r
                break
        
        idx = row_idx * COLS + col
        board[idx] = player
        
        # 4. Check Win
        pos, mask = make_bitboard(board, player)
        if connected_four(pos):
            return history, player # Return history and winner
            
        # 5. Check Draw (Full Board)
        if 0 not in board:
            return history, 0 # 0 means draw
            
        # Switch Player
        player = 2 if player == 1 else 1

def generate_dataset():
    dataset = []
    
    print(f"Generating {NUM_GAMES} games...")
    
    for _ in tqdm(range(NUM_GAMES)):
        # Clear Transposition table occasionally to free memory
        if len(TRANS_TABLE) > 100000:
            TRANS_TABLE.clear()
            
        history, winner = play_game()
        
        # Backpropagate Values
        # If winner is 1:
        #   States where P1 played -> Value = 1
        #   States where P2 played -> Value = -1
        # If Draw (0): All 0
        
        for state, player_turn in history:
            value = 0
            if winner == 1:
                value = 1 if player_turn == 1 else -1
            elif winner == 2:
                value = 1 if player_turn == 2 else -1
            
            # Format: (Board List, Value)
            # You can also add 'move_taken' here if you want Policy Head training
            dataset.append({
                'board': state,
                'value': value
            })
            
    return dataset

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_t = time.time()
    data = generate_dataset()
    
    print(f"Dataset generated with {len(data)} positions.")
    print(f"Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Done! Took {time.time() - start_t:.2f} seconds.")