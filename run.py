#!/usr/bin/env python3
"""
Quick run script for Credit Scoring System

This script provides easy access to common operations.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_training():
    """Run model training."""
    print("🚀 Starting model training...")
    cmd = "python3 scripts/train_models.py"
    subprocess.run(cmd, shell=True)

def run_demo():
    """Run Streamlit demo."""
    print("🌐 Starting Streamlit demo...")
    cmd = "streamlit run demo/streamlit_app.py"
    subprocess.run(cmd, shell=True)

def run_tests():
    """Run tests."""
    print("🧪 Running tests...")
    cmd = "pytest tests/ -v"
    subprocess.run(cmd, shell=True)

def run_setup():
    """Run setup."""
    print("⚙️ Running setup...")
    cmd = "python3 setup.py"
    subprocess.run(cmd, shell=True)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Credit Scoring System Runner")
    parser.add_argument(
        "command",
        choices=["setup", "train", "demo", "test", "all"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    print("🏦 Credit Scoring System")
    print("=" * 30)
    
    if args.command == "setup":
        run_setup()
    elif args.command == "train":
        run_training()
    elif args.command == "demo":
        run_demo()
    elif args.command == "test":
        run_tests()
    elif args.command == "all":
        print("Running full pipeline...")
        run_setup()
        run_training()
        run_tests()
        print("\n✅ Full pipeline completed!")
        print("Run 'python3 run.py demo' to start the web interface")

if __name__ == "__main__":
    main()
