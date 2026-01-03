"""
Credit Scoring System

A comprehensive machine learning system for credit risk assessment and loan approval prediction,
designed for research and educational purposes.

This package provides:
- Data generation and preprocessing for credit scoring
- Multiple machine learning models (Decision Tree, Random Forest, XGBoost, LightGBM, Logistic Regression)
- Comprehensive evaluation metrics including credit-specific metrics
- Interactive web interface for demonstration
- Utility functions for model management and analysis

IMPORTANT: This system is for research and educational purposes only.
It should NOT be used for actual lending decisions or investment advice.
"""

__version__ = "1.0.0"
__author__ = "Credit Scoring System Team"
__email__ = "research@creditscoring.edu"

# Import main modules
from . import data
from . import models
from . import evaluation
from . import utils

__all__ = ['data', 'models', 'evaluation', 'utils']
