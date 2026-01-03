"""
Credit Scoring System - Utility Functions

This module contains utility functions for the credit scoring system.
"""

import os
import logging
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import yaml


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {e}")


def save_model(model: Any, file_path: Union[str, Path], method: str = "joblib") -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model to save
        file_path: Path to save the model
        method: Method to use ('joblib' or 'pickle')
        
    Raises:
        ValueError: If method is not supported
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == "joblib":
        joblib.dump(model, file_path)
    elif method == "pickle":
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported save method: {method}")


def load_model(file_path: Union[str, Path], method: str = "joblib") -> Any:
    """Load a trained model from disk.
    
    Args:
        file_path: Path to the model file
        method: Method to use ('joblib' or 'pickle')
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If method is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    if method == "joblib":
        return joblib.load(file_path)
    elif method == "pickle":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported load method: {method}")


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set additional seeds if libraries are available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """Validate a DataFrame for common issues.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for all-NaN columns
    nan_columns = df.columns[df.isnull().all()].tolist()
    if nan_columns:
        raise ValueError(f"Columns with all NaN values: {nan_columns}")
    
    return True


def get_feature_importance_dict(model: Any, feature_names: list) -> Optional[Dict[str, float]]:
    """Extract feature importance from a model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature importance or None if not available
    """
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, np.abs(model.coef_[0])))
    else:
        return None


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format a number as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a number as percentage.
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Add categorical summaries
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return summary


def print_model_summary(model: Any, model_name: str = "Model") -> None:
    """Print a summary of a trained model.
    
    Args:
        model: Trained model
        model_name: Name of the model
    """
    print(f"\n{model_name} Summary:")
    print("=" * 50)
    print(f"Model Type: {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"Parameters: {len(params)}")
        for key, value in list(params.items())[:5]:  # Show first 5 parameters
            print(f"  {key}: {value}")
        if len(params) > 5:
            print(f"  ... and {len(params) - 5} more parameters")
    
    if hasattr(model, 'feature_importances_'):
        print(f"Feature Importance: Available ({len(model.feature_importances_)} features)")
    elif hasattr(model, 'coef_'):
        print(f"Coefficients: Available ({len(model.coef_[0])} features)")
    else:
        print("Feature Importance: Not available")


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and print duration."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} completed in {duration:.2f} seconds")


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and dependencies.
    
    Returns:
        Dictionary of system information
    """
    import sys
    import platform
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'node': platform.node(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version()
    }
    
    # Check for key packages
    packages = ['numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm', 'matplotlib', 'seaborn']
    package_versions = {}
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            package_versions[package] = version
        except ImportError:
            package_versions[package] = 'Not installed'
    
    system_info['packages'] = package_versions
    
    return system_info
