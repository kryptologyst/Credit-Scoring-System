"""
Credit Scoring System - Data Module

This module contains data generation and processing functionality.
"""

from .credit_data import CreditDataGenerator, CreditDataProcessor, create_train_test_split

__all__ = ['CreditDataGenerator', 'CreditDataProcessor', 'create_train_test_split']
