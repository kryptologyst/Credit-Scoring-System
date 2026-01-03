#!/usr/bin/env python3
"""
Setup script for Credit Scoring System

This script helps set up the environment and test the system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🏦 Credit Scoring System Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("❌ Python 3.10 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Test imports
    print("\nTesting imports...")
    try:
        sys.path.append(str(Path.cwd() / "src"))
        from src.data.credit_data import CreditDataGenerator
        from src.models.credit_models import get_model
        from src.evaluation.credit_evaluator import CreditScoringEvaluator
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Run basic test
    print("\nRunning basic functionality test...")
    try:
        # Generate sample data
        generator = CreditDataGenerator(random_seed=42)
        df = generator.generate_synthetic_data(n_samples=100)
        print(f"✅ Generated {len(df)} samples")
        
        # Test model creation
        model = get_model('decision_tree')
        print("✅ Model creation successful")
        
        print("✅ Basic functionality test passed")
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    # Create necessary directories
    directories = ['data', 'models', 'assets', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python3 scripts/train_models.py")
    print("2. Launch demo: streamlit run demo/streamlit_app.py")
    print("3. Run tests: pytest tests/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
