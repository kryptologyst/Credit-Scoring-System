"""
Tests for Credit Scoring System

This module contains unit tests for the credit scoring system components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.credit_data import CreditDataGenerator, CreditDataProcessor, create_train_test_split
from models.credit_models import (
    BaseCreditModel, DecisionTreeCreditModel, RandomForestCreditModel,
    XGBoostCreditModel, LightGBMCreditModel, LogisticRegressionCreditModel,
    get_model
)
from evaluation.credit_evaluator import CreditScoringEvaluator


class TestCreditDataGenerator:
    """Test cases for CreditDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = CreditDataGenerator(random_seed=42)
        assert generator.random_seed == 42
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        generator = CreditDataGenerator(random_seed=42)
        df = generator.generate_synthetic_data(n_samples=100)
        
        # Check basic properties
        assert len(df) == 100
        assert 'credit_approved' in df.columns
        assert df['credit_approved'].dtype == int
        assert df['credit_approved'].isin([0, 1]).all()
        
        # Check feature columns
        expected_features = [
            'credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio',
            'age', 'years_employed', 'previous_default', 'employment_status',
            'education_level', 'marital_status'
        ]
        for feature in expected_features:
            assert feature in df.columns
    
    def test_realistic_labels(self):
        """Test that labels are generated realistically."""
        generator = CreditDataGenerator(random_seed=42)
        df = generator.generate_synthetic_data(n_samples=1000)
        
        # Check that approval rate is reasonable (not too extreme)
        approval_rate = df['credit_approved'].mean()
        assert 0.1 < approval_rate < 0.9
        
        # Check that high credit scores tend to get approved
        high_score_df = df[df['credit_score'] >= 750]
        if len(high_score_df) > 0:
            high_score_approval_rate = high_score_df['credit_approved'].mean()
            assert high_score_approval_rate > 0.5


class TestCreditDataProcessor:
    """Test cases for CreditDataProcessor."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = CreditDataProcessor()
        assert processor.scaler is not None
        assert processor.label_encoders == {}
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create sample data
        data = {
            'credit_score': [700, 650, 750],
            'annual_income': [60000, 50000, 80000],
            'loan_amount': [25000, 20000, 30000],
            'debt_to_income_ratio': [0.3, 0.4, 0.2],
            'age': [35, 28, 42],
            'years_employed': [5, 3, 10],
            'previous_default': [0, 1, 0],
            'employment_status': ['Employed', 'Unemployed', 'Employed'],
            'education_level': ['Bachelor', 'High School', 'Master'],
            'marital_status': ['Married', 'Single', 'Married'],
            'credit_approved': [1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        processor = CreditDataProcessor()
        X, y = processor.preprocess_data(df)
        
        # Check outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 3
        
        # Check that categorical variables are encoded
        assert X['employment_status'].dtype in [int, np.int64]
        assert X['education_level'].dtype in [int, np.int64]
        assert X['marital_status'].dtype in [int, np.int64]
        
        # Check that numerical variables are scaled
        assert np.isclose(X['credit_score'].std(), 1.0, atol=0.1)
        assert np.isclose(X['annual_income'].std(), 1.0, atol=0.1)
    
    def test_transform_new_data(self):
        """Test transforming new data."""
        # First fit the processor
        data = {
            'credit_score': [700, 650],
            'employment_status': ['Employed', 'Unemployed'],
            'credit_approved': [1, 0]
        }
        df = pd.DataFrame(data)
        
        processor = CreditDataProcessor()
        processor.preprocess_data(df, categorical_columns=['employment_status'])
        
        # Transform new data
        new_data = pd.DataFrame({
            'credit_score': [750],
            'employment_status': ['Employed']
        })
        
        transformed = processor.transform_new_data(new_data)
        assert len(transformed) == 1
        assert 'employment_status' in transformed.columns


class TestCreditModels:
    """Test cases for credit scoring models."""
    
    def test_base_model_interface(self):
        """Test base model interface."""
        model = BaseCreditModel()
        assert model.random_state == 42
        assert not model.is_fitted
        
        # Test that fit raises NotImplementedError
        with pytest.raises(NotImplementedError):
            model.fit(pd.DataFrame(), pd.Series())
    
    def test_decision_tree_model(self):
        """Test Decision Tree model."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([1, 0, 1, 0])
        
        model = DecisionTreeCreditModel(random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.model is not None
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype == int
        
        # Test probability prediction
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (4, 2)
        assert np.isclose(probabilities.sum(axis=1), 1.0).all()
    
    def test_random_forest_model(self):
        """Test Random Forest model."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([1, 0, 1, 0, 1])
        
        model = RandomForestCreditModel(random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([1, 0, 1, 0])
        
        model = XGBoostCreditModel(random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_lightgbm_model(self):
        """Test LightGBM model."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([1, 0, 1, 0])
        
        model = LightGBMCreditModel(random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_logistic_regression_model(self):
        """Test Logistic Regression model."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([1, 0, 1, 0])
        
        model = LogisticRegressionCreditModel(random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_get_model_factory(self):
        """Test model factory function."""
        # Test valid model names
        valid_models = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
        
        for model_name in valid_models:
            model = get_model(model_name)
            assert isinstance(model, BaseCreditModel)
        
        # Test invalid model name
        with pytest.raises(ValueError):
            get_model('invalid_model')


class TestCreditScoringEvaluator:
    """Test cases for CreditScoringEvaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = CreditScoringEvaluator()
        assert evaluator.results == {}
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        y_test = pd.Series([1, 0, 1])
        
        evaluator = CreditScoringEvaluator()
        results = evaluator.evaluate_model(mock_model, X_test, y_test, model_name="test_model")
        
        # Check results structure
        assert 'model_name' in results
        assert 'basic_metrics' in results
        assert 'credit_metrics' in results
        assert 'predictions' in results
        
        # Check basic metrics
        basic_metrics = results['basic_metrics']
        assert 'accuracy' in basic_metrics
        assert 'precision' in basic_metrics
        assert 'recall' in basic_metrics
        assert 'f1_score' in basic_metrics
        assert 'auc_roc' in basic_metrics
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        evaluator = CreditScoringEvaluator()
        
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        y_pred_proba = np.array([0.8, 0.2, 0.7, 0.6])
        
        metrics = evaluator._calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        
        # Check that metrics are valid
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1
    
    def test_create_evaluation_report(self):
        """Test evaluation report creation."""
        # Create mock results
        evaluator = CreditScoringEvaluator()
        evaluator.results['test_model'] = {
            'model_name': 'test_model',
            'basic_metrics': {
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.7,
                'f1_score': 0.72,
                'auc_roc': 0.85,
                'auc_pr': 0.78
            },
            'credit_metrics': {
                'gini_coefficient': 0.7,
                'ks_statistic': 0.4,
                'population_stability_index': 0.1
            },
            'feature_importance': {
                'feature1': 0.6,
                'feature2': 0.4
            }
        }
        
        report = evaluator.create_evaluation_report('test_model')
        
        assert isinstance(report, str)
        assert 'test_model' in report
        assert 'Accuracy:' in report
        assert 'Gini Coefficient:' in report


class TestDataSplits:
    """Test cases for data splitting functions."""
    
    def test_create_train_test_split(self):
        """Test train/validation/test split creation."""
        X = pd.DataFrame({
            'feature1': range(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
            X, y, test_size=0.2, validation_size=0.1, random_state=42
        )
        
        # Check sizes
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        
        # Check that test size is approximately correct
        assert abs(len(X_test) / len(X) - 0.2) < 0.05
        
        # Check that validation size is approximately correct
        val_ratio = len(X_val) / (len(X_train) + len(X_val))
        assert abs(val_ratio - 0.1) < 0.05


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        generator = CreditDataGenerator(random_seed=42)
        df = generator.generate_synthetic_data(n_samples=100)
        
        # Preprocess data
        processor = CreditDataProcessor()
        X, y = processor.preprocess_data(df)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
            X, y, test_size=0.2, validation_size=0.1
        )
        
        # Train model
        model = DecisionTreeCreditModel(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        evaluator = CreditScoringEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test, model_name="test")
        
        # Check that everything worked
        assert model.is_fitted
        assert 'test' in evaluator.results
        assert results['basic_metrics']['accuracy'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
