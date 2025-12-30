import math
import numpy as np
import torch

# --- MCTS CONFIG ---
C_PUCT = 1.414 # Exploration constant
SIMULATIONS = 800 # Number of simulations per move

class Node:
    def __init__(self, board, parent=None, move=None, player=1):
        self.board = board
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.is_terminal = False
        self.winner = 0 # 0=None, 1=P1, -1=P2
    
    def uct_score(self, total_parent_visits, player_perspective):
        if self.visits == 0:
            return float('inf')
        
        # Average value of this node
        q_value = self.value_sum / self.visits

        # Exploration term
        u_value = C_PUCT * math.sqrt(math.log(total_parent_visits) / self.visits)

        # If it's Player 1's turn, they want Maximize Q.
        # If it's Player -1's turn, they want Minimize Q.
        # We adjust Q so "Higher is always better" for the selection formula.
        if player_perspective == -1:
            q_value = -q_value
        
        return q_value + u_value
    
    def is_fully_expanded(self):
        # In this simple version, we just check if we have children.
        # A more complex version tracks all legal moves vs expanded moves.
        return len(self.children) > 0
    
    def best_child(self):
        # Pick the child with the highest visit count
        # This is the final decision logic, not the xploration logic
        return max(self.children, key=lambda node: node.visits)
    
# --- GAME LOGIC HELPERS ---
ROW_COUNT = 6
COLUMN_COUNT = 7

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if board[0][col] == 0:
            valid_locations.append(col)
    return valid_locations

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r

def check_win(board, piece):
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
    # Diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

# --- Core MCTS Function ---
def mcts_search(root_board, model, device):
    # Initialize Root
    # AI is -1. If it's AI's turn to move, the root represents the board BEFORE the move.
    root = Node(root_board, player=1)

    for _ in range(SIMULATIONS):
        node = root

        # 1. Selection
        # Walk down the tree until we find a leaf or an unexpanded node
        while node.is_fully_expanded() and not node.is_terminal:
            # Select child with best UCT score
            node = max(node.children, key=lambda c: c.uct_score(node.visits, node.player))
        
        # 2. Expansion
        # If not terminal, expand one new child
        if not node.is_fully_expanded() and not node.is_terminal:
            valid_moves = get_valid_locations(node.board)

            # Simple approach: Expand all children at once if not expanded
            for col in valid_moves:
                row = get_next_open_row(node.board, col)
                new_board = node.board.copy()
                new_board[row][col] = node.player

                new_node = Node(new_board, parent=node, move=col, player=-node.player)

                # Check for instant win
                if check_win(new_board, node.player):
                    new_node.is_terminal = True
                    new_node.winner = node.player
                
                node.children.append(new_node)
            
            # If we just expanded, pick the first child to simulate/evaluate
            if len(node.children) > 0:
                node = node.children[0]
        
        # 3. Simulation / Evaluation
        value = 0
        if node.is_terminal:
            # If terminal, assign value based on winner
            if node.winner == 1:
                value = 1.0  # Player 1 wins
            elif node.winner == -1:
                value = -1.0 # Player -1 wins
            else:
                value = 0.0  # Draw
        else:
            # Use the neural network to evaluate the board
            tensor = torch.tensor(node.board, dtype=torch.float32).reshape(1, 1, ROW_COUNT, COLUMN_COUNT).to(device)
            with torch.no_grad():
                value = model(tensor).item()  # Assuming model outputs a single scalar value
        
        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent
        
    # Return the move that was visited the most
    if len(root.children) == 0:
        return np.random.choice(get_valid_locations(root_board))
    
    best_node = root.best_child()
    
    return best_node.move