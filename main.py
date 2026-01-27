#!/usr/bin/env python
"""
VegeSSL: Semi-Supervised Contrastive Learning for Vegetation Mislabel Detection

Main entry point for running the complete pipeline:
1. Stage 1: Pixel-level contrastive learning
2. Stage 2: Hard negative mining and encoder refinement
3. Stage 3: Segment-level mislabel detection
4. Visualization: Generate analysis plots

Usage:
    python main.py --stage all              # Run complete pipeline
    python main.py --stage train            # Run training only (Stage 1 & 2)
    python main.py --stage detect           # Run detection only (Stage 3)
    python main.py --stage visualize        # Generate visualizations
    python main.py --config custom.yaml     # Use custom configuration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config


def run_stage1(config_path):
    """Run Stage 1: Pixel-level contrastive learning."""
    print("\n" + "=" * 70)
    print("STAGE 1: PIXEL-LEVEL CONTRASTIVE LEARNING")
    print("=" * 70)
    
    script = PROJECT_ROOT / "experiments" / "train_stage1.py"
    cmd = [sys.executable, str(script)]
    if config_path:
        cmd.extend(["--config", config_path])
    
    subprocess.run(cmd, check=True)


def run_stage2(config_path):
    """Run Stage 2: Hard negative mining."""
    print("\n" + "=" * 70)
    print("STAGE 2: HARD NEGATIVE MINING & REFINEMENT")
    print("=" * 70)
    
    script = PROJECT_ROOT / "experiments" / "train_stage2.py"
    cmd = [sys.executable, str(script)]
    if config_path:
        cmd.extend(["--config", config_path])
    
    subprocess.run(cmd, check=True)


def run_detection(config_path):
    """Run Stage 3: Mislabel detection."""
    print("\n" + "=" * 70)
    print("STAGE 3: SEGMENT-LEVEL MISLABEL DETECTION")
    print("=" * 70)
    
    script = PROJECT_ROOT / "evaluation" / "detect_errors.py"
    cmd = [sys.executable, str(script)]
    if config_path:
        cmd.extend(["--config", config_path])
    
    subprocess.run(cmd, check=True)


def run_visualization(config_path):
    """Generate analysis visualizations."""
    print("\n" + "=" * 70)
    print("VISUALIZATION: GENERATING ANALYSIS PLOTS")
    print("=" * 70)
    
    script = PROJECT_ROOT / "evaluation" / "visualize.py"
    cmd = [sys.executable, str(script)]
    if config_path:
        cmd.extend(["--config", config_path])
    
    subprocess.run(cmd, check=True)


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   VegeSSL: Semi-Supervised Contrastive Learning for             ║
    ║   Vegetation Mislabel Detection in Remote Sensing Data          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="VegeSSL: Mislabel Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --stage all              Run complete pipeline
    python main.py --stage train            Run Stage 1 & 2 only
    python main.py --stage detect           Run Stage 3 detection only
    python main.py --stage visualize        Generate analysis plots
    python main.py --stage 1                Run Stage 1 only
    python main.py --stage 2                Run Stage 2 only
    python main.py --config custom.yaml     Use custom configuration
        """
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "train", "1", "2", "detect", "visualize"],
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/default.yaml)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate configuration
    config_path = args.config
    if config_path:
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        print(f"Using configuration: {config_path}")
    else:
        print("Using default configuration")
    
    # Load config to validate and show info
    config = load_config(config_path)
    print(f"\nData paths:")
    print(f"  Train images: {config.data.train_image_path}")
    print(f"  Test images: {config.data.test_image_path}")
    print(f"  Output: {config.output.base_dir}")
    
    # Run requested stages
    try:
        if args.stage == "all":
            run_stage1(config_path)
            run_stage2(config_path)
            run_detection(config_path)
            run_visualization(config_path)
            
        elif args.stage == "train":
            run_stage1(config_path)
            run_stage2(config_path)
            
        elif args.stage == "1":
            run_stage1(config_path)
            
        elif args.stage == "2":
            run_stage2(config_path)
            
        elif args.stage == "detect":
            run_detection(config_path)
            
        elif args.stage == "visualize":
            run_visualization(config_path)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutputs saved to: {config.output.base_dir}")
        print("\nNext steps:")
        print("  1. Review detected errors in output/tables/")
        print("  2. Examine visualizations in output/figures/")
        print("  3. Adjust thresholds in config as needed")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running pipeline: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
