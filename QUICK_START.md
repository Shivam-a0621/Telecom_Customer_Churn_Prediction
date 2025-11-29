# ⚡ Quick Start Guide - Churn Prediction UI

Get the prediction app running in 3 simple steps!

## Step 1️⃣: Train Models (5 minutes)

Open `churn_prediction.ipynb` and run all cells. This trains 5 different models on your data.

## Step 2️⃣: Save Models (30 seconds)

Add this cell **at the end** of the notebook and run it:

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

# Save feature names
feature_info = {'feature_names': X_train_balanced.columns.tolist()}
with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f)

print('✓ Models saved!')
```

## Step 3️⃣: Launch App (instant)

```bash
# Install dependencies (first time only)
pip install -r requirements_app.txt

# Run the app
streamlit run app.py
```

That's it! The app will open automatically in your browser.

---

## 🎮 Using the App

1. **Pick a Model** - Select from sidebar (Ensemble recommended)
2. **Enter Customer Data** - Fill in 3 tabs with customer information
3. **Click Predict** - Get instant churn risk assessment
4. **Review Results** - See probability, risk category, and recommendations

## 📊 What You Get

✅ Churn probability (0-100%)
✅ Risk category (🔴 High / 🟡 Medium / 🟢 Low)
✅ Personalized recommendations
✅ Business impact analysis
✅ Comparison from all 5 models

## 🆘 Having Issues?

**App won't start?**
```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib
```

**Models not found?**
- Make sure `models/` directory exists
- Check models were saved to `models/*.pkl`
- Verify `models/feature_info.json` exists

**Error during prediction?**
- Check input values are reasonable
- Try refreshing the browser
- Try a different model from dropdown

## 📚 Learn More

See `README_APP.md` for detailed documentation.

---

**Happy predicting! 🎯**
