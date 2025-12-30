import time
import math

def my_agent(observation, configuration):
    # Constant and Config
    ROWS = configuration.rows
    COLS = configuration.columns
    # This bitboard implementation is optimized for 6x7 but works for sizes up to 7x9
    # or 8x8 as long as (ROWS + 1) * COLS <= 64

    # Time limit buffer
    TIME_LIMIT = 2.0

    # Memoization/Transposition Table
    # We use global dictionary to persist data across different calls of 'agent'
    global TRANS_TABLE
    if 'TRANS_TABLE' not in globals():
        TRANS_TABLE = {}
    

    # Bitboard Helpers
    def make_bitboard(board_list, player_mark):
        """Converts Kaggle's board list into two bitboards: (position, mask)"""
        position = 0
        mask = 0
        # Kaggle board is row-major. We need to convert it to our bitboard format.
        for c in range(COLS):
            for r in range(ROWS):
                # Calculate the bit index
                idx = r * COLS + c
                if board_list[idx] != 0:
                    mask |= (1 << (c * (ROWS + 1) + (ROWS - 1 - r)))
                    if board_list[idx] == player_mark:
                        position |= (1 << (c * (ROWS + 1) + (ROWS - 1 - r)))
        return position, mask
    
    def get_valid_moves(mask):
        """Returns a list of valid columns where a piece can be dropped."""
        moves = []
        center = COLS // 2
        column_order = [center]
        for i in range(1, COLS):
            if center - i >= 0:
                column_order.append(center - i)
            if center + i < COLS:
                column_order.append(center + i)
        
        for col in column_order:
            # Check if the top cell of the column is empty
            if (mask & (1 << (col * (ROWS + 1) + ROWS - 1))) == 0:
                moves.append(col)
        return moves
    
    def connected_four(position):
        """Check if a specific player has won using bitshifts."""
        # Horizontal check
        m = position & (position >> (ROWS + 1))
        if m & (m >> (2 * (ROWS + 1))):
            return True
        # Diagonal /
        m = position & (position >> ROWS)
        if m & (m >> (2 * ROWS)):
            return True
        # Diagonal \
        m = position & (position >> (ROWS + 2))
        if m & (m >> (2 * (ROWS + 2))):
            return True
        # Vertical check
        m = position & (position >> 1)
        if m & (m >> 2):
            return True
        return False
    
    def make_move(position, mask, col):
        """Returns new (position, mask) after making a move in the specified column."""
        new_position = position ^ mask
        new_mask = mask | (mask + (1 << (col * (ROWS + 1))))
        return new_position, new_mask
    
    # --- Search Algorithm (NEGAMAX) ---
    def negamax(position, mask, depth, alpha, beta, start_time):
        # Check for timeout
        if time.time() - start_time > TIME_LIMIT:
            return None
        
        # Check Transposition Table
        key = (position, mask)
        if key in TRANS_TABLE and TRANS_TABLE[key][1] >= depth:
            return TRANS_TABLE[key][0]
        
        # Check for immediate win/loss
        # Note: In negamax, we check if the PREVIOUS player has won
        # Since we just flipped perspective, 'position' is now the current player and 'position ^ mask' is the opponent
        oppenent_position = position ^ mask
        if connected_four(oppenent_position):
            return -(10000 + depth) # Prefer winning sooner
        
        if depth == 0:
            return 0 # Draw or neutral evaluation at depth 0
        
        # If board is full
        if mask == ((1 << ((ROWS + 1) * COLS)) - 1) ^ ((1 << ((ROWS + 1) * COLS)) // ((1 << (ROWS + 1)) - 1)):
            return 0
        
        valid_moves = get_valid_moves(mask)

        best_score = -math.inf

        for col in valid_moves:
            # Play move and swap perspective
            # The new position is the opponent's old position (mask ^ position)
            # plus the new piece. But simpler:
            # logic is: new_pos = current_pos ^ mask (this flips to opponent view)
            # new_mask includes the new piece.
            new_pos = position ^ mask
            mew_mask = mask | (mask + (1 << (col * (ROWS + 1))))

            score = negamax(new_pos, mew_mask, depth - 1, -beta, -alpha, start_time)

            # Timeout propagation
            if score is None:
                return None
            
            score = -score

            if score > best_score:
                best_score = score
            
            alpha = max(alpha, score)
            if alpha >= beta:
                break # Pruning

        # Store in Transposition Table
        TRANS_TABLE[key] = (best_score, depth)
        return best_score
    
    # --- Main Agent Logic ---
    start_time = time.time()

    # 1. Setup Bitboards
    # player_mark is 1 or 2.
    my_mark = observation.mark
    position, mask = make_bitboard(observation.board, my_mark)

    # 2. Get Valid Moves
    valid_moves = get_valid_moves(mask)
    if not valid_moves:
        return 0 # No valid moves, should not happen in normal gameplay
    
    # 3. Iterative Deepening Negamax
    best_move = valid_moves[0] # Default fallback
    depth = 1

    # If there is only one move, dont waste time searching
    if len(valid_moves) == 1:
        return valid_moves[0]
    
    while True:
        current_best_move = None
        max_score = -math.inf

        # Search this depth
        for col in valid_moves:
            new_pos = position ^ mask
            new_mask = mask | (mask + (1 << (col * (ROWS + 1))))

            # Call negamax
            score = negamax(new_pos, new_mask, depth, -math.inf, math.inf, start_time)

            if score is None:
                break 

            score = -score

            if score > max_score:
                max_score = score
                current_best_move = col
        
        # If we timed out during the loop, break and use previous best depth result
        if score is None:
            break

        best_move = current_best_move

        # If we found a guaranteed win, no need to search deeper
        if max_score >= 9000:
            break

        depth += 1
        # Safety break to prevent infinite loop in case of unexpected behavior
        if depth > 42:
            break

    return best_move