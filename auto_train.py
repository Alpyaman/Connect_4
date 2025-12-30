import self_play
import retrain
import time
import os
import shutil
import torch
from model import Connect4Net
import pickle
import random
from collections import deque

# --- CONFIG ---
ITERATIONS = 20          # Increase to 20 to see long-term stability
GAMES_PER_LOOP = 50      # Generate 50 new games
MAX_BUFFER_SIZE = 5000   # Keep the last 5000 games in memory
CHECKPOINT_DIR = "checkpoints"
BUFFER_FILE = "replay_buffer.pkl"

def save_buffer(buffer):
    with open(BUFFER_FILE, 'wb') as f:
        pickle.dump(buffer, f)

def load_buffer():
    if os.path.exists(BUFFER_FILE):
        with open(BUFFER_FILE, 'rb') as f:
            return pickle.load(f)
    return deque(maxlen=MAX_BUFFER_SIZE)

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    print("Starting Auto-Train with Replay Buffer...")
    
    # Initialize Buffer
    replay_buffer = load_buffer()
    print(f"Replay Buffer loaded with {len(replay_buffer)} games.")

    # Save Gen 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_gen_0.pth"))
    
    start_time = time.time()
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n>>> CYCLE {i}/{ITERATIONS}")
        
        # 1. GENERATE NEW DATA
        self_play.GAMES_TO_PLAY = GAMES_PER_LOOP
        # We output to a temp file so we can ingest it manually
        self_play.OUTPUT_FILE = "temp_new_data.pkl" 
        self_play.generate_dataset()
        
        # 2. UPDATE REPLAY BUFFER
        with open("temp_new_data.pkl", 'rb') as f:
            new_data = pickle.load(f)
            
        # Add new games to buffer
        # new_data is a list of dicts: {'board': ..., 'value': ...}
        for item in new_data:
            replay_buffer.append(item)
            
        print(f"Buffer updated. Total samples: {len(replay_buffer)}")
        save_buffer(replay_buffer)
        
        # 3. PREPARE TRAINING DATA
        # We save the ENTIRE buffer (or a random subset) to 'self_play_data.pkl' for retrain.py
        # To prevent overfitting, let's train on a random sample if buffer is huge
        training_data = list(replay_buffer)
        if len(training_data) > 2000:
             # Train on latest 500 + random 1500 (Mix Recency + Diversity)
             recent = list(replay_buffer)[-500:]
             older = random.sample(list(replay_buffer)[:-500], 1500)
             training_data = recent + older
             
        with open("self_play_data.pkl", 'wb') as f:
            pickle.dump(training_data, f)
            
        print(f"Training on {len(training_data)} samples mixed from history.")

        # 4. RETRAIN
        retrain.retrain()
        
        # 5. CHECKPOINT
        current_model = "connect4_cnn.pth"
        checkpoint_name = os.path.join(CHECKPOINT_DIR, f"model_gen_{i}.pth")
        shutil.copy(current_model, checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")

    print(f"\nDone! Total time: {(time.time() - start_time)/60:.1f} min")

if __name__ == "__main__":
    main()