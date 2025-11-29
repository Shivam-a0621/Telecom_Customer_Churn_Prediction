"""
Customer Churn Prediction Streamlit App
========================================
Based on: churn_prediction.ipynb
Features from telecom customer dataset with engineered features
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= LOAD MODELS & METADATA =============
@st.cache_resource
def load_models():
    """Load all saved models and metadata"""
    models_dir = Path("saved_models")

    if not models_dir.exists():
        return None, None, None

    try:
        models = {
            'Baseline (Logistic Regression)': joblib.load(models_dir / 'baseline_model.pkl'),
            'Random Forest': joblib.load(models_dir / 'rf_improved.pkl'),
            'XGBoost': joblib.load(models_dir / 'xgb_improved.pkl'),
            'Logistic Regression': joblib.load(models_dir / 'lr_improved.pkl'),
            'Ensemble (Voting)': joblib.load(models_dir / 'ensemble_model.pkl')
        }
        
        # Load metadata
        metadata = joblib.load(models_dir / 'feature_metadata.pkl')
        
        # Load state columns
        state_columns = joblib.load(models_dir / 'state_columns.pkl')

        return models, metadata, state_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load resources
models, metadata, state_columns = load_models()
models_loaded = models is not None

# ============= STREAMLIT APP =============
st.title("📊 Customer Churn Prediction")
st.markdown("---")

if models_loaded:
    # Model selection in sidebar
    st.sidebar.header("⚙️ Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Prediction Model",
        options=sorted(list(models.keys())),
        help="Choose which trained model to use for prediction"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Descriptions")
    st.sidebar.markdown("""
    - **Ensemble**: Voting classifier (RF + XGB + LR)
    - **XGBoost**: Gradient boosting model
    - **Random Forest**: Tree-based ensemble
    - **Logistic Regression**: Linear classifier
    - **Baseline**: Baseline LR model
    """)

    # Display selected model
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📌 Using model: **{selected_model.upper()}**")
    with col2:
        st.caption(f"Total Models: {len(models)}")
    
    st.markdown("---")
    st.subheader("📋 Customer Information")
    
    st.info("""
    **💡 Tip:** Churn is more likely for customers with:
    - **Short account length** (< 10 months)
    - **Many service calls** (> 5 calls)
    - **Low usage** (few minutes/month)
    - **High charges** relative to usage
    
    Try entering different values to see how risk changes!
    """)

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📍 Basic Info", "📞 Usage Metrics", "💼 Plans & Services"])

    user_input = {}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Location & Account")
            # Extract state list from metadata
            all_features = metadata['all_features']
            state_features = [f for f in all_features if f.startswith('State_')]
            states = sorted([f.replace('State_', '') for f in state_features])
            
            user_input['State'] = st.selectbox("State", states, index=0)
            user_input['Area code'] = st.number_input("Area Code", 200, 999, 415)
            user_input['Account length'] = st.slider("Account Length (months)", 0, 250, 120)

        with col2:
            st.markdown("### Plans")
            user_input['International plan'] = st.radio(
                "International Plan",
                ['No', 'Yes'],
                help="Does customer have international plan?"
            )
            user_input['Voice mail plan'] = st.radio(
                "Voice Mail Plan",
                ['No', 'Yes'],
                help="Does customer have voice mail plan?"
            )
            user_input['Number vmail messages'] = st.number_input(
                "Voice Mail Messages",
                0, 100, 25
            )

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Day Usage")
            user_input['Total day minutes'] = st.number_input(
                "Day Minutes", 0.0, 1000.0, 200.0, step=1.0
            )
            user_input['Total day calls'] = st.number_input(
                "Day Calls", 0, 200, 100, step=1
            )
            user_input['Total day charge'] = st.number_input(
                "Day Charge ($)", 0.0, 300.0, 35.0, step=0.01
            )

        with col2:
            st.markdown("### Evening Usage")
            user_input['Total eve minutes'] = st.number_input(
                "Evening Minutes", 0.0, 1000.0, 180.0, step=1.0
            )
            user_input['Total eve calls'] = st.number_input(
                "Evening Calls", 0, 200, 90, step=1
            )
            user_input['Total eve charge'] = st.number_input(
                "Evening Charge ($)", 0.0, 300.0, 15.0, step=0.01
            )

        with col3:
            st.markdown("### Night & International")
            user_input['Total night minutes'] = st.number_input(
                "Night Minutes", 0.0, 1000.0, 200.0, step=1.0
            )
            user_input['Total night calls'] = st.number_input(
                "Night Calls", 0, 200, 95, step=1
            )
            user_input['Total night charge'] = st.number_input(
                "Night Charge ($)", 0.0, 300.0, 9.0, step=0.01
            )
            user_input['Total intl minutes'] = st.number_input(
                "Intl Minutes", 0.0, 100.0, 10.0, step=0.1
            )
            user_input['Total intl calls'] = st.number_input(
                "Intl Calls", 0, 50, 5, step=1
            )
            user_input['Total intl charge'] = st.number_input(
                "Intl Charge ($)", 0.0, 30.0, 2.7, step=0.01
            )

    with tab3:
        st.markdown("### Customer Service")
        user_input['Customer service calls'] = st.slider(
            "Customer Service Calls",
            0, 10, 1,
            help="More calls may indicate dissatisfaction"
        )
    
    st.markdown("---")
    
    # ============= PREDICTION SECTION =============
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Churn Risk",
            type="primary",
            use_container_width=True
        )

    if predict_button:
        try:
            # Create feature vector from user input
            feature_dict = {}

            # Add all basic features
            for key in ['Account length', 'Area code', 'Number vmail messages',
                       'Total day minutes', 'Total day calls', 'Total day charge',
                       'Total eve minutes', 'Total eve calls', 'Total eve charge',
                       'Total night minutes', 'Total night calls', 'Total night charge',
                       'Total intl minutes', 'Total intl calls', 'Total intl charge',
                       'Customer service calls']:
                feature_dict[key] = user_input.get(key, 0)

            # Convert categorical to numeric
            feature_dict['International plan'] = 1 if user_input['International plan'] == 'Yes' else 0
            feature_dict['Voice mail plan'] = 1 if user_input['Voice mail plan'] == 'Yes' else 0

            # ============= CREATE ENGINEERED FEATURES (same as notebook) =============
            # Total usage minutes
            feature_dict['total_usage_minutes'] = (
                user_input['Total day minutes'] + 
                user_input['Total eve minutes'] + 
                user_input['Total night minutes'] + 
                user_input['Total intl minutes']
            )
            
            # Average call duration
            total_calls = (
                user_input['Total day calls'] + 
                user_input['Total eve calls'] + 
                user_input['Total night calls'] + 
                user_input['Total intl calls'] + 1  # +1 to avoid division by zero
            )
            feature_dict['avg_call_duration'] = feature_dict['total_usage_minutes'] / total_calls
            
            # Day to night ratio
            feature_dict['day_night_ratio'] = (
                user_input['Total day minutes'] / 
                (user_input['Total night minutes'] + 1)  # +1 to avoid division by zero
            )
            
            # Service call risk (inverse relationship)
            feature_dict['service_call_risk'] = 1 / (user_input['Customer service calls'] + 1)
            
            # Has international service
            feature_dict['has_intl_service'] = 1 if user_input['Total intl minutes'] > 0 else 0

            # One-hot encode State
            selected_state = user_input['State']
            # Get all states from metadata
            all_features = metadata['all_features']
            state_features = [f for f in all_features if f.startswith('State_')]
            states = [f.replace('State_', '') for f in state_features]

            # Add state one-hot encoding
            for state in states:
                feature_dict[f'State_{state}'] = 1 if state == selected_state else 0

            # Create DataFrame with proper feature names to suppress warnings
            input_df = pd.DataFrame([feature_dict])
            
            # Reorder columns to match training data order
            input_df = input_df[metadata['all_features']]

            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0, 1]
            
            # ============= DISPLAY RESULTS =============
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            # Determine risk category
            if probability > 0.7:
                risk_level = "🔴 HIGH RISK"
                risk_color = "error"
                recommendation = "Immediate action required"
            elif probability > 0.4:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = "warning"
                recommendation = "Monitor & nurture"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "success"
                recommendation = "Standard service"

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}",
                    delta=None
                )

            with col2:
                st.metric(
                    "Prediction",
                    "CHURN" if prediction == 1 else "NO CHURN",
                    delta=None
                )

            with col3:
                st.metric(
                    "Risk Category",
                    risk_level.replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', ''),
                    delta=None
                )

            # Display risk box
            if risk_color == "error":
                st.error(f"**{risk_level}**\n{recommendation}")
            elif risk_color == "warning":
                st.warning(f"**{risk_level}**\n{recommendation}")
            else:
                st.success(f"**{risk_level}**\n{recommendation}")
            
            # ============= DETAILED ANALYSIS =============
            st.markdown("---")
            st.subheader("💡 Customer Profile & Recommendations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Customer Characteristics:**")
                profile = f"""
                - **State:** {user_input['State']}
                - **Account Length:** {user_input['Account length']} months
                - **Area Code:** {user_input['Area code']}
                - **International Plan:** {user_input['International plan']}
                - **Voice Mail Plan:** {user_input['Voice mail plan']}
                - **Total Usage:** {user_input['Total day minutes'] + user_input['Total eve minutes'] + user_input['Total night minutes']:.0f} min/month
                - **Customer Service Calls:** {user_input['Customer service calls']}
                """
                st.markdown(profile)

            with col2:
                st.markdown("**Recommended Actions:**")
                if probability > 0.7:
                    actions = """
                    🔴 **CRITICAL - Immediate Response**
                    - Contact customer within 24 hours
                    - Offer personalized retention incentives
                    - Address service issues directly
                    - Assign dedicated account manager
                    - Consider service upgrade or discount
                    """
                elif probability > 0.4:
                    actions = """
                    🟡 **HIGH PRIORITY - Monitor Closely**
                    - Schedule regular check-ins
                    - Offer value-added services
                    - Gather feedback on satisfaction
                    - Provide loyalty rewards
                    - Review service quality
                    """
                else:
                    actions = """
                    🟢 **STANDARD - Maintain Relationship**
                    - Continue regular communication
                    - Provide quality service
                    - Offer loyalty incentives
                    - Encourage referrals
                    - Periodic satisfaction surveys
                    """
                st.markdown(actions)

            # ============= BUSINESS IMPACT =============
            st.markdown("---")
            st.subheader("💰 Business Impact Analysis")

            # Calculate potential impact
            avg_monthly_charge = 10
            avg_lifespan = 36
            cltv = avg_monthly_charge * avg_lifespan
            retention_cost = 50
            success_rate = 0.75

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Customer Lifetime Value", f"${cltv:.0f}")
            with col2:
                st.metric("Retention Cost", f"${retention_cost:.0f}")
            with col3:
                revenue_at_risk = cltv if prediction == 1 else 0
                st.metric("Revenue at Risk", f"${revenue_at_risk:.0f}")
            with col4:
                revenue_saveable = cltv * success_rate if prediction == 1 else 0
                st.metric("Potential Savings", f"${revenue_saveable:.0f}")

            # ============= MODEL COMPARISON =============
            st.markdown("---")
            st.subheader("🔄 All Models Predictions")

            comparison_data = []
            for model_name in sorted(models.keys()):
                model_obj = models[model_name]
                pred = model_obj.predict(input_df)[0]
                prob = model_obj.predict_proba(input_df)[0, 1]
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Prediction': 'CHURN' if pred == 1 else 'NO CHURN',
                    'Probability': f"{prob:.1%}",
                })

            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.caption("**Note:** Comparing predictions across different models can provide confidence in the result.")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("**Troubleshooting:**\n1. Make sure all models are saved in the 'models' directory\n2. Check that feature_info.json exists\n3. Ensure all required libraries are installed")

    st.markdown("---")

else:
    st.error("❌ Models Not Found")
    st.markdown("""
    ### Setup Instructions

    Before running predictions, you need to train and save the models:

    1. **Run the training notebook** (`churn_prediction.ipynb`)
    2. **Save the models** by adding this code cell at the end of the notebook:

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

    print('Setup complete! All models saved.')
    ```

    3. **Refresh this app** - the models will be automatically loaded

    ### What Models Are Available?
    - **Ensemble**: Voting classifier combining RF, XGB, and LR
    - **XGBoost**: Gradient boosting model (Best F1: 0.8369)
    - **Random Forest**: Tree-based ensemble
    - **Logistic Regression**: Linear classifier
    - **Baseline**: Simple baseline model for comparison

    ### About the Data
    - **Dataset**: Telecom customer churn dataset (2,666 customers)
    - **Features**: 73 engineered features including usage metrics, plans, and derived features
    - **Target**: Customer churn (Yes/No)
    - **Class Balance**: SMOTE applied for imbalanced classes
    """)

# Footer
st.markdown("---")
st.markdown("""
### 📊 About This Application
Built to predict customer churn and identify at-risk customers for proactive retention.

**Key Features:**
- Compare predictions across 5 different models
- Get personalized risk assessment and recommendations
- Calculate business impact and ROI
- Track customer characteristics and usage patterns

**Disclaimer:** Predictions are based on historical patterns and trained models. Always combine predictions with business judgment and customer context for best results.
""")
