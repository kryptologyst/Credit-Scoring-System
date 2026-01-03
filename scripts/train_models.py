"""
Credit Scoring System - Main Training Script

This script demonstrates the complete credit scoring pipeline including
data generation, model training, evaluation, and comparison.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.credit_data import CreditDataGenerator, CreditDataProcessor, create_train_test_split
from src.models.credit_models import get_model
from src.evaluation.credit_evaluator import CreditScoringEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_and_evaluate_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train and evaluate multiple credit scoring models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Starting credit scoring model training and evaluation")
    
    # Set random seeds for reproducibility
    np.random.seed(config.get('data', {}).get('synthetic', {}).get('random_seed', 42))
    
    # Generate synthetic data
    logger.info("Generating synthetic credit data")
    data_generator = CreditDataGenerator(random_seed=42)
    df = data_generator.generate_synthetic_data(
        n_samples=config.get('data', {}).get('synthetic', {}).get('n_samples', 10000)
    )
    
    # Preprocess data
    logger.info("Preprocessing data")
    processor = CreditDataProcessor()
    X, y = processor.preprocess_data(df)
    
    # Create train/validation/test splits
    logger.info("Creating data splits")
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
        X, y,
        test_size=config.get('data', {}).get('synthetic', {}).get('test_size', 0.2),
        validation_size=config.get('data', {}).get('synthetic', {}).get('validation_size', 0.1)
    )
    
    # Initialize evaluator
    evaluator = CreditScoringEvaluator()
    
    # Define models to train
    models_to_train = [
        'decision_tree',
        'random_forest', 
        'xgboost',
        'lightgbm',
        'logistic_regression'
    ]
    
    # Train and evaluate each model
    results = {}
    trained_models = {}
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name} model")
        
        # Get model configuration
        model_config = config.get('model', {}).get(model_name, {})
        
        # Create and train model
        model = get_model(model_name, **model_config)
        model.fit(X_train, y_train)
        
        # Evaluate model
        evaluation_results = evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name
        )
        
        results[model_name] = evaluation_results
        trained_models[model_name] = model
        
        # Print evaluation report
        print(evaluator.create_evaluation_report(model_name))
        print("\n" + "="*80 + "\n")
    
    return results, trained_models, processor


def compare_models(results: Dict[str, Any]) -> pd.DataFrame:
    """Compare performance of all models.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        DataFrame with model comparison
    """
    logger.info("Creating model comparison")
    
    comparison_data = []
    
    for model_name, result in results.items():
        basic_metrics = result['basic_metrics']
        credit_metrics = result['credit_metrics']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': basic_metrics['accuracy'],
            'Precision': basic_metrics['precision'],
            'Recall': basic_metrics['recall'],
            'F1-Score': basic_metrics['f1_score'],
            'AUC-ROC': basic_metrics['auc_roc'],
            'AUC-PR': basic_metrics['auc_pr'],
            'Gini': credit_metrics['gini_coefficient'],
            'KS Statistic': credit_metrics['ks_statistic']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
    
    return comparison_df


def create_model_comparison_plots(results: Dict[str, Any], save_dir: str = "assets"):
    """Create comparison plots for all models.
    
    Args:
        results: Dictionary of evaluation results
        save_dir: Directory to save plots
    """
    logger.info("Creating model comparison plots")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics for comparison
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model]['basic_metrics'][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create ROC curves comparison
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        y_true = result['predictions']['y_true']
        y_pred_proba = result['predictions']['y_pred_proba']
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = result['basic_metrics']['auc_roc']
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the credit scoring pipeline."""
    logger.info("Starting Credit Scoring System")
    
    # Load configuration
    config = load_config()
    
    # Train and evaluate models
    results, trained_models, processor = train_and_evaluate_models(config)
    
    # Compare models
    comparison_df = compare_models(results)
    print("\nModel Comparison:")
    print("="*80)
    print(comparison_df.round(4))
    
    # Save comparison results
    os.makedirs("assets", exist_ok=True)
    comparison_df.to_csv("assets/model_comparison.csv", index=False)
    
    # Create comparison plots
    create_model_comparison_plots(results)
    
    # Save trained models (optional)
    import joblib
    os.makedirs("models", exist_ok=True)
    
    for model_name, model in trained_models.items():
        joblib.dump(model, f"models/{model_name}_model.pkl")
    
    joblib.dump(processor, "models/data_processor.pkl")
    
    logger.info("Credit scoring pipeline completed successfully")
    
    # Print best model
    best_model = comparison_df.iloc[0]['Model']
    best_auc = comparison_df.iloc[0]['AUC-ROC']
    print(f"\nBest performing model: {best_model} (AUC-ROC: {best_auc:.4f})")


if __name__ == "__main__":
    main()
