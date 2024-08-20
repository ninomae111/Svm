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
feature_names = ['CCB', 'Last bowel movement was clear liquid', 'Split dose', 'In hospital bowel preparation', 
    'Bowel movement status', 'Activity level', 'Education']

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
features = [ccb, last_bowel_movement_was_clear_liquid, split_dose, in_hospital_bowel_preparation, 
    bowel_movement_status, activity_level, education]
feature_values = pd.DataFrame([features])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(feature_values)[0]
    predicted_proba = model.predict_proba(feature_values)[0]

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
    
    # 创建 SHAP KernelExplainer
    explainer = shap.KernelExplainer(model.predict_proba, feature_values)
    shap_values = explainer.shap_values(feature_values)

    # 显示两个类别的 SHAP 力图
    st.write("### SHAP Force Plot for Class 0")
    shap.force_plot(explainer.expected_value[0], shap_values[0], feature_values, feature_names=feature_names)

    st.write("### SHAP Force Plot for Class 1")
    shap.force_plot(explainer.expected_value[1], shap_values[1], feature_values, feature_names=feature_names)

# 假设 k = 0，因为只有一个样本
k = 0
if features is None:
    display_features = ["" for i in range(len(feature_names))]
else:
    # 检查features的形状，确保不超过索引
    if features.shape[0] > k:
        display_features = features[k, :]
    else:
        display_features = features[0, :]  # 或者你可以调整索引逻辑，确保使用有效索引

# 使用st_shap函数来显示图像
def st_shap(plot, height=None):
    """Helper function to display SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], feature_values, feature_names=feature_names))
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], feature_values, feature_names=feature_names))
