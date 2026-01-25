import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def run_baselines():
    """Run baseline comparisons."""
    baselines = [
        "MobileNetV2", "DenseNet121", "ResNet50V2", 
        "EfficientNetV2B0", "VGG16"
    ]
    
    print("Running Baseline Comparisons...")
    
    for model_name in baselines:
        print(f"\n--- Training Baseline: {model_name} ---")
        # Train baseline model
        # Save metrics
        
    print("\nBaseline Comparison Complete!")

if __name__ == "__main__":
    run_baselines()
