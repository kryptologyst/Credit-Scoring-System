"""
Credit Scoring System - Data Generation and Processing Module

This module handles data generation, preprocessing, and feature engineering
for the credit scoring system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class CreditDataGenerator:
    """Generate synthetic credit scoring data for demonstration purposes."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic credit scoring dataset.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic credit data
        """
        logger.info(f"Generating {n_samples} synthetic credit samples")
        
        # Generate base features
        data = {
            'credit_score': np.random.randint(300, 850, n_samples),
            'annual_income': np.random.normal(60000, 20000, n_samples),
            'loan_amount': np.random.normal(25000, 10000, n_samples),
            'debt_to_income_ratio': np.random.uniform(0.05, 0.50, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'years_employed': np.random.randint(0, 50, n_samples),
            'previous_default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-employed'], 
                                                n_samples, p=[0.7, 0.15, 0.15]),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                               n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                              n_samples, p=[0.4, 0.45, 0.15])
        }
        
        df = pd.DataFrame(data)
        
        # Ensure positive values for income and loan amount
        df['annual_income'] = np.abs(df['annual_income'])
        df['loan_amount'] = np.abs(df['loan_amount'])
        
        # Generate realistic credit approval based on features
        df['credit_approved'] = self._generate_realistic_labels(df)
        
        logger.info(f"Generated dataset with shape: {df.shape}")
        logger.info(f"Credit approval rate: {df['credit_approved'].mean():.3f}")
        
        return df
    
    def _generate_realistic_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic credit approval labels based on features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of credit approval labels
        """
        # Create a more realistic approval logic
        approval_prob = np.zeros(len(df))
        
        # Credit score factor (most important)
        approval_prob += np.where(df['credit_score'] >= 700, 0.4, 
                                 np.where(df['credit_score'] >= 650, 0.2, 
                                         np.where(df['credit_score'] >= 600, 0.1, 0.05)))
        
        # Income factor
        approval_prob += np.where(df['annual_income'] >= 80000, 0.2,
                                 np.where(df['annual_income'] >= 50000, 0.1, 0.05))
        
        # Debt-to-income ratio (lower is better)
        approval_prob += np.where(df['debt_to_income_ratio'] <= 0.2, 0.15,
                                 np.where(df['debt_to_income_ratio'] <= 0.35, 0.1, 0.05))
        
        # Employment status
        employment_factor = df['employment_status'].map({
            'Employed': 0.1, 'Self-employed': 0.05, 'Unemployed': -0.1
        })
        approval_prob += employment_factor
        
        # Previous default (negative factor)
        approval_prob += np.where(df['previous_default'] == 0, 0.1, -0.2)
        
        # Age factor (middle age preferred)
        age_factor = np.where((df['age'] >= 25) & (df['age'] <= 55), 0.05, 0.02)
        approval_prob += age_factor
        
        # Add some randomness
        approval_prob += np.random.normal(0, 0.1, len(df))
        
        # Convert to binary labels
        return (approval_prob > 0.5).astype(int)


class CreditDataProcessor:
    """Process and preprocess credit scoring data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       target_column: str = 'credit_approved',
                       categorical_columns: Optional[List[str]] = None,
                       numerical_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the credit data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Tuple of (processed_features, target)
        """
        logger.info("Preprocessing credit data")
        
        # Set default columns if not provided
        if categorical_columns is None:
            categorical_columns = ['employment_status', 'education_level', 'marital_status']
        if numerical_columns is None:
            numerical_columns = ['credit_score', 'annual_income', 'loan_amount', 
                               'debt_to_income_ratio', 'age', 'years_employed', 'previous_default']
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        # Separate features and target
        feature_columns = categorical_columns + numerical_columns
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        X_scaled = X.copy()
        X_scaled[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        self.feature_columns = X_scaled.columns.tolist()
        
        logger.info(f"Preprocessed data shape: {X_scaled.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors.
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        if self.feature_columns is None:
            raise ValueError("Must fit preprocessor first")
        
        X = df[self.feature_columns].copy()
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in X.columns and col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        X[self.numerical_columns] = self.scaler.transform(X[self.numerical_columns])
        
        return X


def create_train_test_split(X: pd.DataFrame, 
                          y: pd.Series, 
                          test_size: float = 0.2,
                          validation_size: float = 0.1,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                         pd.Series, pd.Series, pd.Series]:
    """Create train, validation, and test splits.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        validation_size: Proportion for validation set (from remaining data)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Creating train/validation/test splits")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
