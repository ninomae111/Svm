import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('SVM.pkl')

# Define feature options
ccb_options = {
    0: 'No',
    1: 'Yes'
}

last_bowel_movement_was_clear_liquid_options = {
    0: 'No',
    1: 'Yes'
}

split_dose_options = {
    0: 'No',
    1: 'Yes'
}

in_hospital_bowel_preparation_options = {
    0: 'No',
    1: 'Yes'
}

bowel_movement_status_options = {
    1: 'Normal',
    2: 'Diarrhea',
    3: 'Constipation'
}

activity_level_options = {
    0: 'Frequently walking',
    1: 'Occasionally walking'
}

education_options = {
    1: 'Primary school or below',
    2: 'Middle school',
    3: 'High school or above'
}

# Define feature names
feature_names = [
    'CCB', 
    'Last bowel movement was clear liquid', 
    'Split dose',
    'In hospital bowel preparation', 
    'Bowel movement status', 
    'Activity level', 
    'Education'
]

# Streamlit user interface
st.title("Bowel Preparation Predictor")

# CCB: categorical selection
ccb = st.selectbox("CCB:", options=list(ccb_options.keys()), format_func=lambda x: ccb_options[x])

# Last bowel movement was clear liquid: categorical selection
last_bowel_movement_was_clear_liquid = st.selectbox("Last bowel movement was clear liquid:", options=list(last_bowel_movement_was_clear_liquid_options.keys()), format_func=lambda x: last_bowel_movement_was_clear_liquid_options[x])

# Split-dose: categorical selection
split_dose = st.selectbox("Split dose:", options=list(split_dose_options.keys()), format_func=lambda x: split_dose_options[x])

# In-hospital bowel preparation: categorical selection
in_hospital_bowel_preparation = st.selectbox("In hospital bowel preparation:", options=list(in_hospital_bowel_preparation_options.keys()), format_func=lambda x: in_hospital_bowel_preparation_options[x])

# Bowel movement status: categorical selection
bowel_movement_status = st.selectbox("Bowel movement status:", options=list(bowel_movement_status_options.keys()), format_func=lambda x: bowel_movement_status_options[x])

# Activity level: categorical selection
activity_level = st.selectbox("Activity level:", options=list(activity_level_options.keys()), format_func=lambda x: activity_level_options[x])

# Education: categorical selection
education = st.selectbox("Education:", options=list(education_options.keys()), format_func=lambda x: education_options[x])

# Process inputs and make predictions
feature_values = np.array([
    ccb, 
    last_bowel_movement_was_clear_liquid, 
    split_dose,
    in_hospital_bowel_preparation, 
    bowel_movement_status, 
    activity_level, 
    education
]).reshape(1, -1)  # 转换为NumPy数组

# Convert features to DataFrame
features = pd.DataFrame(feature_values, columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of poor bowel preparation. "
            f"The model predicts that your probability of poor bowel preparation is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult your attending physician or charge nurse "
            "to achieve better bowel preparation quality."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of poor bowel preparation. "
            f"The model predicts that your probability of good bowel preparation is {probability:.1f}%. "
            "However, it is still very important to follow medical instructions for bowel preparation. "
            "I recommend that you follow the advice of your doctor or charge nurse for bowel preparation."
        )

    st.write(advice)

    # SHAP Explanation (optional)
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(features, nsamples=100))
    shap_values = explainer.shap_values(features)

    # 输出 SHAP 值的形状以进行调试
    st.write(f"SHAP values shape: {np.array(shap_values).shape}")

    # 选择特定类别的 SHAP 值（例如，正类别）
    class_index = 1  # 可以根据需要选择 0 或 1，取决于您想要解释哪个类别
    shap_values_for_class = shap_values[class_index]

    # Check if SHAP values match the feature length
    if shap_values_for_class.shape[1] == len(feature_names):
        shap.force_plot(explainer.expected_value[class_index], shap_values_for_class[0], features.iloc[0], feature_names=feature_names, matplotlib=True)
        st.pyplot(plt.gcf())
    else:
        st.error("Mismatch between feature and SHAP values dimensions.")
