import os

from miner_neat2 import run_neat, TrainingVisualizer
_ = TrainingVisualizer()
OUTPUT_DIR = "output/neat/20250526-204754"

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(
        config_file=config_file,
        output_dir=OUTPUT_DIR,
        continue_from_checkpoint=True
    )
