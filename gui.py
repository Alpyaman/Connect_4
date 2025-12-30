import pygame
import numpy as np
import torch
import math
import sys
import mcts
from model import Connect4Net
import os

# --- CONFIG ---
MODEL_PATH = "checkpoints/model_gen_20.pth" # Load your smartest generation
# If gen_20 doesn't exist, use "connect4_cnn.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "connect4_cnn.pth"

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)

# --- GAME LOGIC ---
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

def winning_move(board, piece):
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

def draw_board(board, screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height - int((ROW_COUNT-1-r)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == -1: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height - int((ROW_COUNT-1-r)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

def main():
    pygame.init()
    screen = pygame.display.set_mode(size)
    myfont = pygame.font.SysFont("monospace", 50)
    
    # Load AI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Loaded AI Brain: {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model. Playing Randomly. Error: {e}")

    board = create_board()
    draw_board(board, screen)
    game_over = False
    turn = 0 # 0 = Player (Red), 1 = AI (Yellow)

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == 0:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                
                # Player 1 Input
                if turn == 0:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, 1)

                        if winning_move(board, 1):
                            label = myfont.render("PLAYER 1 WINS!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn = 1
                        draw_board(board, screen)

        # AI Turn
        if turn == 1 and not game_over:
            # Draw "Thinking..." text
            label = myfont.render("AI Thinking...", 1, YELLOW)
            screen.blit(label, (40, 10))
            pygame.display.update()
            
            # Use MCTS
            col = mcts.mcts_search(board, model, device)
            
            # Clear text
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, -1)

                if winning_move(board, -1):
                    label = myfont.render("AI WINS!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                draw_board(board, screen)
                turn = 0

        if game_over:
            pygame.time.wait(3000)

if __name__ == "__main__":
    main()