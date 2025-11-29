# 📊 Customer Churn Prediction UI

A user-friendly Streamlit web application for predicting customer churn using multiple machine learning models.

## 🎯 Features

- **Interactive Web UI**: Enter customer information and get instant churn predictions
- **Multiple Models**: Compare predictions from 5 different models (Ensemble, XGBoost, Random Forest, Logistic Regression, Baseline)
- **Risk Assessment**: Categorizes customers into High, Medium, and Low risk categories
- **Business Impact Analysis**: Calculates potential revenue at risk and retention ROI
- **Model Comparison**: View predictions from all models side-by-side
- **Actionable Recommendations**: Get specific retention strategies based on risk level

## 📋 Input Features

### Basic Information
- **State**: Customer's state (50 US states)
- **Area Code**: Telephone area code
- **Account Length**: How long the customer has been with the company
- **Plans**: International plan, Voice mail plan status

### Usage Metrics
- **Day Usage**: Total minutes, calls, and charges
- **Evening Usage**: Total minutes, calls, and charges
- **Night Usage**: Total minutes, calls, and charges
- **International Usage**: Total minutes, calls, and charges

### Customer Service
- **Customer Service Calls**: Number of support calls made

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 2. Train Models (One-time setup)

First, run the `churn_prediction.ipynb` notebook to train all models. Then add this code cell at the very end:

```python
import joblib
from pathlib import Path
import json

# Create models directory
Path('models').mkdir(exist_ok=True)

# Save all trained models
models_to_save = {
    'ensemble': ensemble_model,
    'xgboost': xgb_improved,
    'random_forest': rf_improved,
    'logistic_regression': lr_improved,
    'baseline': baseline_model
}

for model_name, model in models_to_save.items():
    joblib.dump(model, f'models/{model_name}_model.pkl')
    print(f'Saved {model_name} model')

# Save feature names
feature_info = {'feature_names': X_train_balanced.columns.tolist()}
with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f)

print('✓ All models saved successfully!')
```

Run this cell to save the trained models.

### 3. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Select a Model**: Choose from the sidebar which model to use for prediction
   - **Ensemble**: Recommended - combines all models
   - **XGBoost**: Best individual performance
   - **Random Forest**: Good for interpretability
   - **Logistic Regression**: Fast and simple
   - **Baseline**: For comparison

2. **Enter Customer Information**:
   - Fill in the 3 tabs with customer data
   - Use realistic values for accurate predictions

3. **Get Prediction**:
   - Click "🔮 Predict Churn Risk" button
   - View immediate risk assessment

4. **Review Results**:
   - **Churn Probability**: Likelihood customer will churn (0-100%)
   - **Prediction**: CHURN or NO CHURN
   - **Risk Category**: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
   - **Recommendations**: Specific actions based on risk level
   - **Business Impact**: Revenue at risk and ROI
   - **Model Comparison**: See all models' predictions

## 📈 Model Performance

From the training notebook:

| Model | F1-Score | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| **XGBoost** | **0.8369** | **0.9569** | 0.8772 |
| Ensemble | 0.7448 | 0.9307 | 0.8738 |
| Random Forest | 0.7059 | 0.9251 | 0.8785 |
| Logistic Regression | 0.4400 | 0.7903 | 0.7507 |
| Baseline | 0.3089 | 0.8408 | 0.7669 |

## 💰 Business Context

- **Dataset**: 2,666 telecom customers
- **Churn Rate**: 14.55%
- **Average Customer Lifetime Value**: $360 (10 months × 36 months)
- **Retention Cost**: $50 per customer
- **Retention Success Rate**: 75%
- **Potential ROI**: Up to 440% on retention investments

## 🔧 Directory Structure

```
pikky_churn/
├── app.py                    # Streamlit application
├── churn_prediction.ipynb     # Training notebook
├── requirements_app.txt       # Python dependencies
├── README_APP.md              # This file
├── models/                    # Directory for saved models
│   ├── ensemble_model.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── baseline_model.pkl
│   └── feature_info.json
└── churn-bigml-80.csv        # Training data
```

## 🎨 Risk Categories

### 🔴 HIGH RISK (Probability > 70%)
- **Immediate action required**
- Contact customer within 24 hours
- Offer personalized retention incentives
- Address service issues
- Assign dedicated account manager

### 🟡 MEDIUM RISK (Probability 40-70%)
- **Monitor and nurture relationship**
- Schedule regular check-ins
- Offer value-added services
- Gather feedback
- Provide loyalty rewards

### 🟢 LOW RISK (Probability < 40%)
- **Standard service**
- Continue regular communication
- Provide quality service
- Include in standard loyalty programs
- Periodic satisfaction surveys

## 🐛 Troubleshooting

### Models Not Loading
- Ensure `models/` directory exists
- Check that all `.pkl` files are present
- Verify `feature_info.json` exists
- Reinstall dependencies: `pip install -r requirements_app.txt`

### Prediction Errors
- Check all input values are within reasonable ranges
- Ensure feature_info.json has correct feature names
- Try a different model from the dropdown

### App Won't Start
- Install Streamlit: `pip install streamlit`
- Check Python version (3.7+)
- Verify app.py is in the current directory

## 📚 Understanding the Prediction

The model considers these key factors:

1. **Usage Patterns**: How much the customer uses the service
2. **Account Tenure**: How long they've been a customer
3. **Service Usage**: Day, evening, night, and international usage
4. **Plans**: Whether they have additional plans
5. **Customer Service Interactions**: Number of support calls

**Higher usage** = More satisfied = Lower churn risk
**More service calls** = More frustrated = Higher churn risk

## 🔐 Data Privacy

- The app runs locally on your machine
- No data is sent to external servers
- No personal information is logged
- All predictions happen in-memory

## 📄 License

This project is based on the telecom customer churn dataset with machine learning models trained for educational and business purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the training notebook for model details
3. Verify all dependencies are installed correctly

---

**Built with Streamlit | Powered by scikit-learn, XGBoost, and ensemble methods**
