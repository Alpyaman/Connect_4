import self_play
import retrain
import time
import os
import shutil
import torch
from model import Connect4Net

# --- CONFIG ---
ITERATIONS = 10         # How many generations to train
GAMES_PER_LOOP = 50     # Games per generation
CHECKPOINT_DIR = "checkpoints"

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    print(f"Starting Auto-Train for {ITERATIONS} cycles...")
    start_time = time.time()
    
    # Save the initial random model as Gen 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_gen_0.pth"))
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n>>> CYCLE {i}/{ITERATIONS}")
        
        # 1. Self Play
        self_play.GAMES_TO_PLAY = GAMES_PER_LOOP
        self_play.OUTPUT_FILE = "self_play_data.pkl" 
        self_play.generate_dataset()
        
        # 2. Retrain
        retrain.retrain()
        
        # 3. Save Checkpoint
        # We copy the just-updated 'connect4_cnn.pth' to the checkpoints folder
        current_model = "connect4_cnn.pth"
        checkpoint_name = os.path.join(CHECKPOINT_DIR, f"model_gen_{i}.pth")
        shutil.copy(current_model, checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")

    print(f"\nDone! Total time: {(time.time() - start_time)/60:.1f} min")

if __name__ == "__main__":
    main()