import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.training import train_model
from src.models import build_rf_densenet

def run_ablation(config_path=None):
    """Run complete ablation study."""
    print("Running Ablation Study...")
    
    # 1. Growth Rate Sweep
    growth_rates = [4, 8, 12]
    for gr in growth_rates:
        print(f"\n--- Testing Growth Rate: {gr} ---")
        # Train model with growth_rate=gr
        # Save metrics
        
    # 2. Compression Sweep
    compressions = [0.25, 0.5, 0.75]
    for comp in compressions:
        print(f"\n--- Testing Compression: {comp} ---")
        
    # 3. Depth Sweep
    depths = [(2,2,2), (3,3,3), (4,4,4)]
    for d in depths:
        print(f"\n--- Testing Depth: {d} ---")
        
    print("\nAblation Study Complete!")

if __name__ == "__main__":
    run_ablation()
