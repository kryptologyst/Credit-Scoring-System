# Credit Scoring System

A comprehensive machine learning system for credit risk assessment and loan approval prediction, designed for research and educational purposes.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration only.**

This credit scoring system is for academic and research purposes. It should NOT be used for:
- Making actual lending decisions
- Investment advice  
- Real-world financial applications

The models may be inaccurate and are based on synthetic data. Always consult with qualified financial professionals for actual credit decisions.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Credit-Scoring-System.git
cd Credit-Scoring-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
python scripts/train_models.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

## Features

### Multiple Machine Learning Models
- **Decision Tree**: Baseline tree-based classifier
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **Logistic Regression**: Linear baseline model

### Comprehensive Evaluation
- Standard ML metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR)
- Credit-specific metrics (Gini Coefficient, KS Statistic, Population Stability Index)
- Cross-validation with stratified sampling
- Feature importance analysis

### Interactive Web Interface
- Real-time credit approval prediction
- Data visualization and analysis
- Model performance comparison
- Risk factor identification

## Project Structure

```
0485_Credit_Scoring_System/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   └── credit_data.py        # Data generation and preprocessing
│   ├── models/                   # Model definitions
│   │   └── credit_models.py      # Credit scoring models
│   ├── evaluation/               # Evaluation metrics
│   │   └── credit_evaluator.py   # Comprehensive evaluator
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── scripts/                      # Training scripts
│   └── train_models.py          # Main training pipeline
├── demo/                         # Interactive demo
│   └── streamlit_app.py         # Streamlit web app
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── assets/                       # Generated outputs
├── models/                       # Trained models
├── data/                         # Data storage
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
data:
  synthetic:
    n_samples: 10000
    test_size: 0.2
    validation_size: 0.1

model:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

evaluation:
  comprehensive:
    metrics: ["accuracy", "precision", "recall", "f1", "auc", "gini"]
    cross_validation:
      cv_folds: 5
```

## Usage Examples

### Training Models

```python
from src.data.credit_data import CreditDataGenerator, CreditDataProcessor
from src.models.credit_models import get_model
from src.evaluation.credit_evaluator import CreditScoringEvaluator

# Generate synthetic data
generator = CreditDataGenerator(random_seed=42)
df = generator.generate_synthetic_data(n_samples=10000)

# Preprocess data
processor = CreditDataProcessor()
X, y = processor.preprocess_data(df)

# Train model
model = get_model('xgboost')
model.fit(X_train, y_train)

# Evaluate model
evaluator = CreditScoringEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
```

### Making Predictions

```python
# New borrower data
borrower_data = {
    'credit_score': 720,
    'annual_income': 75000,
    'loan_amount': 30000,
    'debt_to_income_ratio': 0.25,
    'age': 35,
    'years_employed': 5,
    'previous_default': 0,
    'employment_status': 'Employed',
    'education_level': 'Bachelor',
    'marital_status': 'Married'
}

# Transform and predict
X_new = processor.transform_new_data(pd.DataFrame([borrower_data]))
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)
```

## Model Performance

The system evaluates models using both standard ML metrics and credit-specific metrics:

### Standard Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Credit-Specific Metrics
- **Gini Coefficient**: Model discrimination ability (0-1, higher is better)
- **KS Statistic**: Separation between good and bad borrowers
- **Population Stability Index**: Score distribution stability

## Credit Scoring Features

The system analyzes the following borrower characteristics:

### Financial Information
- **Credit Score**: FICO-style score (300-850)
- **Annual Income**: Household income
- **Loan Amount**: Requested loan amount
- **Debt-to-Income Ratio**: Monthly debt / monthly income

### Personal Information
- **Age**: Borrower's age
- **Years Employed**: Employment tenure
- **Previous Default**: Default history
- **Employment Status**: Current employment
- **Education Level**: Highest education
- **Marital Status**: Marital status

## Data Generation

The system generates realistic synthetic credit data with:

- **Realistic Distributions**: Based on real-world credit patterns
- **Feature Correlations**: Realistic relationships between features
- **Label Generation**: Approval decisions based on multiple factors
- **Class Balance**: Configurable approval rates

## Web Interface

The Streamlit demo provides:

### Model Demo Page
- Interactive borrower information input
- Real-time credit approval prediction
- Probability visualization with gauges
- Risk factor identification
- Model selection

### Data Analysis Page
- Dataset overview and statistics
- Feature distribution analysis
- Correlation matrix visualization
- Interactive plots

### Model Comparison Page
- Performance metrics comparison
- Visualization of model differences
- Best model identification

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis

### Adding New Models

1. Create a new model class inheriting from `BaseCreditModel`
2. Implement the `fit` method
3. Add to the model factory in `credit_models.py`
4. Update configuration files

### Adding New Metrics

1. Implement metric calculation in `CreditScoringEvaluator`
2. Add to the evaluation configuration
3. Update visualization functions

## References

- FICO Score Components
- Basel II/III Credit Risk Framework
- Machine Learning for Credit Risk Assessment
- Fair Lending Regulations

## Contributing

This is a research demonstration project. Contributions are welcome for:
- Additional model implementations
- New evaluation metrics
- Documentation improvements
- Bug fixes

## License

This project is for educational and research purposes only. See the disclaimer above for important usage restrictions.

## Related Projects

- Credit Risk Modeling Frameworks
- Fair Lending Compliance Tools
- Alternative Credit Scoring Methods
- Explainable AI for Financial Services

---

**Remember**: This system is for research and education only. Never use for actual lending decisions or investment advice.
# Credit-Scoring-System
