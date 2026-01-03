"""
Credit Scoring System - Streamlit Demo

Interactive web application for credit scoring demonstration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import joblib
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.credit_data import CreditDataGenerator, CreditDataProcessor
from src.models.credit_models import get_model
from src.evaluation.credit_evaluator import CreditScoringEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This credit scoring system is for academic and research purposes. It should NOT be used for:</p>
    <ul>
        <li>Making actual lending decisions</li>
        <li>Investment advice</li>
        <li>Real-world financial applications</li>
    </ul>
    <p>The models may be inaccurate and are based on synthetic data. Always consult with qualified financial professionals for actual credit decisions.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏦 Credit Scoring System</h1>', unsafe_allow_html=True)
st.markdown("### Research & Educational Demonstration")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Model Demo", "Data Analysis", "Model Comparison", "About"]
)

@st.cache_data
def load_sample_data():
    """Load sample credit data."""
    generator = CreditDataGenerator(random_seed=42)
    return generator.generate_synthetic_data(n_samples=1000)

@st.cache_resource
def load_trained_models():
    """Load pre-trained models."""
    models = {}
    model_names = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
    
    try:
        for model_name in model_names:
            model_path = f"models/{model_name}_model.pkl"
            if Path(model_path).exists():
                models[model_name] = joblib.load(model_path)
        
        if Path("models/data_processor.pkl").exists():
            processor = joblib.load("models/data_processor.pkl")
            return models, processor
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return None, None

def predict_credit_score(features, model, processor):
    """Predict credit score for given features."""
    try:
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Transform features
        transformed_features = processor.transform_new_data(feature_df)
        
        # Make prediction
        prediction = model.predict(transformed_features)[0]
        probability = model.predict_proba(transformed_features)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

if page == "Model Demo":
    st.header("Credit Score Prediction Demo")
    
    # Load models
    models, processor = load_trained_models()
    
    if models is None or processor is None:
        st.error("No trained models found. Please run the training script first.")
        st.info("Run: `python scripts/train_models.py`")
    else:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Borrower Information")
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model:",
                list(models.keys()),
                help="Choose the credit scoring model to use"
            )
            
            # Input fields
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                help="FICO-style credit score"
            )
            
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=500000,
                value=60000,
                step=1000,
                help="Annual household income"
            )
            
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=0,
                max_value=1000000,
                value=25000,
                step=1000,
                help="Requested loan amount"
            )
            
            debt_to_income = st.slider(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Monthly debt payments / monthly income"
            )
            
            age = st.slider(
                "Age",
                min_value=18,
                max_value=80,
                value=35,
                help="Borrower's age"
            )
            
            years_employed = st.slider(
                "Years Employed",
                min_value=0,
                max_value=50,
                value=5,
                help="Years at current job"
            )
            
            previous_default = st.selectbox(
                "Previous Default",
                ["No", "Yes"],
                help="Has the borrower defaulted before?"
            )
            
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-employed", "Unemployed"],
                help="Current employment status"
            )
            
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Bachelor", "Master", "PhD"],
                help="Highest education level"
            )
            
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
                help="Marital status"
            )
            
            # Prepare features
            features = {
                'credit_score': credit_score,
                'annual_income': annual_income,
                'loan_amount': loan_amount,
                'debt_to_income_ratio': debt_to_income,
                'age': age,
                'years_employed': years_employed,
                'previous_default': 1 if previous_default == "Yes" else 0,
                'employment_status': employment_status,
                'education_level': education_level,
                'marital_status': marital_status
            }
            
            # Predict button
            if st.button("Predict Credit Approval", type="primary"):
                model = models[selected_model]
                prediction, probability = predict_credit_score(features, model, processor)
                
                if prediction is not None:
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.probability = probability
                    st.session_state.model_used = selected_model
        
        with col2:
            st.subheader("Prediction Results")
            
            if 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                probability = st.session_state.probability
                model_used = st.session_state.model_used
                
                # Display results
                if prediction == 1:
                    st.success("✅ **CREDIT APPROVED**")
                    st.balloons()
                else:
                    st.error("❌ **CREDIT DENIED**")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Denial Probability", f"{probability[0]:.1%}")
                with col2b:
                    st.metric("Approval Probability", f"{probability[1]:.1%}")
                
                # Model info
                st.info(f"Model used: {model_used.replace('_', ' ').title()}")
                
                # Risk factors
                st.subheader("Risk Assessment")
                risk_factors = []
                
                if credit_score < 600:
                    risk_factors.append("Low credit score")
                if debt_to_income > 0.4:
                    risk_factors.append("High debt-to-income ratio")
                if previous_default == "Yes":
                    risk_factors.append("Previous default")
                if employment_status == "Unemployed":
                    risk_factors.append("Unemployment")
                if years_employed < 2:
                    risk_factors.append("Short employment history")
                
                if risk_factors:
                    st.warning("Risk factors identified:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No major risk factors identified")
            else:
                st.info("👈 Fill in the borrower information and click 'Predict Credit Approval' to see results")

elif page == "Data Analysis":
    st.header("Credit Data Analysis")
    
    # Load sample data
    df = load_sample_data()
    
    st.subheader("Dataset Overview")
    st.write(f"**Dataset size:** {len(df):,} records")
    st.write(f"**Features:** {len(df.columns)-1}")
    st.write(f"**Approval rate:** {df['credit_approved'].mean():.1%}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    # Select feature to analyze
    feature_cols = [col for col in df.columns if col != 'credit_approved']
    selected_feature = st.selectbox("Select feature to analyze:", feature_cols)
    
    # Create distribution plot
    fig = px.histogram(
        df, 
        x=selected_feature, 
        color='credit_approved',
        title=f"Distribution of {selected_feature} by Credit Approval",
        labels={'credit_approved': 'Credit Approved'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlations")
    
    # Calculate correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    # Check if comparison results exist
    comparison_path = Path("assets/model_comparison.csv")
    
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        
        st.subheader("Performance Metrics")
        st.dataframe(comparison_df.round(4))
        
        # Create comparison charts
        st.subheader("Performance Visualization")
        
        # Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = comparison_df.loc[comparison_df['AUC-ROC'].idxmax()]
        st.success(f"🏆 **Best Model:** {best_model['Model']} (AUC-ROC: {best_model['AUC-ROC']:.4f})")
        
    else:
        st.warning("No model comparison results found. Please run the training script first.")
        st.info("Run: `python scripts/train_models.py`")

elif page == "About":
    st.header("About the Credit Scoring System")
    
    st.markdown("""
    ### Overview
    This Credit Scoring System is a comprehensive machine learning application designed for 
    research and educational purposes. It demonstrates various techniques used in credit risk 
    assessment and loan approval processes.
    
    ### Features
    - **Multiple Models**: Decision Tree, Random Forest, XGBoost, LightGBM, and Logistic Regression
    - **Comprehensive Evaluation**: Standard ML metrics plus credit-specific metrics (Gini, KS statistic)
    - **Interactive Demo**: Real-time credit approval prediction
    - **Data Analysis**: Feature distributions and correlation analysis
    - **Model Comparison**: Performance benchmarking across different algorithms
    
    ### Technical Stack
    - **Python 3.10+**
    - **Machine Learning**: scikit-learn, XGBoost, LightGBM
    - **Data Processing**: pandas, numpy
    - **Visualization**: plotly, matplotlib, seaborn
    - **Web Interface**: Streamlit
    
    ### Credit-Specific Metrics
    - **Gini Coefficient**: Measures model discrimination ability
    - **Kolmogorov-Smirnov Statistic**: Measures separation between good and bad borrowers
    - **Population Stability Index**: Measures score distribution stability
    
    ### Model Features
    - Credit Score (300-850)
    - Annual Income
    - Loan Amount
    - Debt-to-Income Ratio
    - Age
    - Years Employed
    - Previous Default History
    - Employment Status
    - Education Level
    - Marital Status
    
    ### Usage
    1. **Model Demo**: Input borrower information to get credit approval prediction
    2. **Data Analysis**: Explore the synthetic dataset and feature relationships
    3. **Model Comparison**: Compare performance across different algorithms
    
    ### Important Notes
    - This system uses synthetic data for demonstration purposes
    - Models are not trained on real credit data
    - Results should not be used for actual lending decisions
    - Always consult with qualified financial professionals for real-world applications
    """)
    
    st.subheader("Contact & Support")
    st.markdown("""
    This is a research demonstration project. For questions or issues:
    - Check the documentation in the repository
    - Review the training scripts for implementation details
    - Ensure all dependencies are properly installed
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Credit Scoring System - Research & Educational Demonstration Only<br>
    Not for use in actual lending decisions or investment advice
</div>
""", unsafe_allow_html=True)
