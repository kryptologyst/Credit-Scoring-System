"""
Credit Scoring System - Evaluation Module

This module provides comprehensive evaluation metrics and analysis for credit scoring models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class CreditScoringEvaluator:
    """Comprehensive evaluator for credit scoring models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def evaluate_model(self, 
                      model: Any,
                      X_test: pd.DataFrame, 
                      y_test: pd.Series,
                      X_train: Optional[pd.DataFrame] = None,
                      y_train: Optional[pd.Series] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """Evaluate a credit scoring model comprehensively.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (for cross-validation)
            y_train: Training targets (for cross-validation)
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        metrics = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        
        # Credit-specific metrics
        credit_metrics = self._calculate_credit_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation if training data provided
        cv_scores = None
        if X_train is not None and y_train is not None:
            cv_scores = self._cross_validate_model(model, X_train, y_train)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.columns)
        
        results = {
            'model_name': model_name,
            'basic_metrics': metrics,
            'credit_metrics': credit_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'predictions': {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        self.results[model_name] = results
        logger.info(f"Evaluation completed for {model_name}")
        
        return results
    
    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': self._calculate_auc_pr(y_true, y_pred_proba)
        }
    
    def _calculate_credit_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate credit-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of credit metrics
        """
        # Gini coefficient
        gini = 2 * roc_auc_score(y_true, y_pred_proba) - 1
        
        # Kolmogorov-Smirnov statistic
        ks_stat = self._calculate_ks_statistic(y_true, y_pred_proba)
        
        # Population Stability Index (simplified)
        psi = self._calculate_psi(y_pred_proba)
        
        return {
            'gini_coefficient': gini,
            'ks_statistic': ks_stat,
            'population_stability_index': psi
        }
    
    def _calculate_auc_pr(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            AUC-PR score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return np.trapz(precision, recall)
    
    def _calculate_ks_statistic(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            KS statistic
        """
        # Separate probabilities by class
        prob_good = y_pred_proba[y_true == 0]
        prob_bad = y_pred_proba[y_true == 1]
        
        # Calculate cumulative distributions
        thresholds = np.linspace(0, 1, 100)
        ks_values = []
        
        for threshold in thresholds:
            cum_good = np.mean(prob_good <= threshold)
            cum_bad = np.mean(prob_bad <= threshold)
            ks_values.append(abs(cum_good - cum_bad))
        
        return max(ks_values)
    
    def _calculate_psi(self, y_pred_proba: np.ndarray) -> float:
        """Calculate Population Stability Index (simplified version).
        
        Args:
            y_pred_proba: Predicted probabilities
            
        Returns:
            PSI value
        """
        # For simplicity, we'll use a basic PSI calculation
        # In practice, you'd compare against a reference distribution
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(y_pred_proba, bins=bins)
        hist = hist / len(y_pred_proba)
        
        # Avoid division by zero
        hist = np.where(hist == 0, 1e-6, hist)
        
        # Calculate PSI (simplified)
        psi = np.sum(hist * np.log(hist / (1/len(hist))))
        return psi
    
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                             cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Note: We need to create a new model instance for each fold
        # This is a simplified version - in practice, you'd handle this differently
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc_roc': []
        }
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create new model instance (this is simplified)
            fold_model = type(model)(**model.__dict__)
            fold_model.fit(X_train_fold, y_train_fold)
            
            y_pred_fold = fold_model.predict(X_val_fold)
            y_pred_proba_fold = fold_model.predict_proba(X_val_fold)[:, 1]
            
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred_fold))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred_fold))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred_fold))
            cv_scores['auc_roc'].append(roc_auc_score(y_val_fold, y_pred_proba_fold))
        
        return cv_scores
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance or None
        """
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance = dict(zip(feature_names, np.abs(model.coef_[0])))
        
        return importance
    
    def create_evaluation_report(self, model_name: str) -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        results = self.results[model_name]
        
        report = f"""
Credit Scoring Model Evaluation Report
=====================================
Model: {model_name}

Basic Metrics:
--------------
Accuracy:  {results['basic_metrics']['accuracy']:.4f}
Precision: {results['basic_metrics']['precision']:.4f}
Recall:    {results['basic_metrics']['recall']:.4f}
F1-Score:  {results['basic_metrics']['f1_score']:.4f}
AUC-ROC:   {results['basic_metrics']['auc_roc']:.4f}
AUC-PR:    {results['basic_metrics']['auc_pr']:.4f}

Credit-Specific Metrics:
------------------------
Gini Coefficient:        {results['credit_metrics']['gini_coefficient']:.4f}
KS Statistic:           {results['credit_metrics']['ks_statistic']:.4f}
Population Stability:    {results['credit_metrics']['population_stability_index']:.4f}

Cross-Validation Results:
------------------------
"""
        
        if results['cv_scores']:
            cv_scores = results['cv_scores']
            for metric, scores in cv_scores.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                report += f"{metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})\n"
        
        if results['feature_importance']:
            report += "\nTop 5 Most Important Features:\n"
            report += "-----------------------------\n"
            sorted_features = sorted(results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                report += f"{feature}: {importance:.4f}\n"
        
        return report
    
    def plot_evaluation_metrics(self, model_name: str, save_path: Optional[str] = None):
        """Plot evaluation metrics and visualizations.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save plots
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        results = self.results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(results['predictions']['y_true'], 
                               results['predictions']['y_pred_proba'])
        axes[0, 0].plot(fpr, tpr, label=f"AUC = {results['basic_metrics']['auc_roc']:.3f}")
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(results['predictions']['y_true'],
                                                     results['predictions']['y_pred_proba'])
        axes[0, 1].plot(recall, precision, label=f"AUC-PR = {results['basic_metrics']['auc_pr']:.3f}")
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        
        # Confusion Matrix
        cm = confusion_matrix(results['predictions']['y_true'], results['predictions']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Feature Importance
        if results['feature_importance']:
            features = list(results['feature_importance'].keys())
            importance = list(results['feature_importance'].values())
            sorted_idx = np.argsort(importance)[::-1][:10]  # Top 10 features
            
            axes[1, 1].barh(range(len(sorted_idx)), [importance[i] for i in sorted_idx])
            axes[1, 1].set_yticks(range(len(sorted_idx)))
            axes[1, 1].set_yticklabels([features[i] for i in sorted_idx])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
