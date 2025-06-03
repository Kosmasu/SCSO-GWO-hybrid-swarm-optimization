import os
import glob
from datetime import datetime
from miner_neat2 import run_neat, TrainingVisualizer

def find_latest_checkpoint_dir():
    """Find the most recent training output directory."""
    output_pattern = "output/neat/????????-??????"  # Pattern: YYYYMMDD-HHMMSS
    dirs = glob.glob(output_pattern)
    
    if not dirs:
        print("❌ No training directories found in output/neat/")
        return None
    
    # Sort by creation time (most recent first)
    dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    latest_dir = dirs[0]
    
    print(f"🔍 Found {len(dirs)} training directories")
    print(f"📁 Latest: {latest_dir}")
    
    # Verify it has checkpoints
    checkpoints_dir = os.path.join(latest_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        print(f"❌ No checkpoints directory found in {latest_dir}")
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("neat-checkpoint-")]
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoints_dir}")
        return None
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint files")
    return latest_dir

def continue_training(output_dir=None):
    """Continue training from the specified or latest checkpoint directory."""
    
    if output_dir is None:
        output_dir = find_latest_checkpoint_dir()
        if output_dir is None:
            print("❌ Cannot continue training - no valid checkpoint found")
            return
    else:
        # Verify the specified directory exists and has checkpoints
        if not os.path.exists(output_dir):
            print(f"❌ Directory not found: {output_dir}")
            return
        
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            print(f"❌ No checkpoints directory found in {output_dir}")
            return
        
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("neat-checkpoint-")]
        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {checkpoints_dir}")
            return
    
    print(f"\n🔄 Continuing training from: {output_dir}")
    
    # Get config file path
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return
    
    # Start training
    print(f"📋 Using config: {config_file}")
    print(f"⏰ Resuming at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    run_neat(
        config_file=config_file,
        output_dir=output_dir,
        continue_from_checkpoint=True
    )

if __name__ == "__main__":
    # Option 1: Continue from specific directory
    SPECIFIC_OUTPUT_DIR = "output/neat/20250603-163922"
    
    # Option 2: Auto-find latest directory (set to None)
    # SPECIFIC_OUTPUT_DIR = None
    
    # Check if specific directory exists
    if SPECIFIC_OUTPUT_DIR and os.path.exists(SPECIFIC_OUTPUT_DIR):
        print(f"📁 Using specified directory: {SPECIFIC_OUTPUT_DIR}")
        continue_training(SPECIFIC_OUTPUT_DIR)
    else:
        print("🔍 Specified directory not found, searching for latest...")
        continue_training(None)  # Auto-find latest