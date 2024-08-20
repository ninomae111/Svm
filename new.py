import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import log_loss, accuracy_score

# Load data
df = pd.read_csv('建模 - 副本240610.csv')

# Split features and target variable
X = df.drop(['Fail'], axis=1)
y = df['Fail']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72, stratify=y)

# Initialize the SVM classifier with fixed hyperparameters
model_svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)

# Bootstrap resampling and model training
n_iterations = 1000
log_loss_list = []
accuracy_list = []

for i in range(n_iterations):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    model_svm.fit(X_resampled, y_resampled)
    
    y_pred_proba = model_svm.predict_proba(X_resampled)
    y_pred = model_svm.predict(X_resampled)
    
    log_loss_value = log_loss(y_resampled, y_pred_proba)
    accuracy_value = accuracy_score(y_resampled, y_pred)
    
    log_loss_list.append(log_loss_value)
    accuracy_list.append(accuracy_value)

# Calculate mean and standard deviation of log loss and accuracy
mean_log_loss = np.mean(log_loss_list)
std_log_loss = np.std(log_loss_list)
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

# Print the results
print(f"Mean Log Loss: {mean_log_loss:.4f} (± {std_log_loss:.4f})")
print(f"Mean Accuracy: {mean_accuracy:.4f} (± {std_accuracy:.4f})")

# Set the trained model
model = model_svm

# Define feature options
use_calcium_channel_blockers_options = {0: 'No', 1: 'Yes'}
last_bowel_movement_was_clear_liquid_options = {0: 'No', 1: 'Yes'}
split_dose_options = {0: 'No', 1: 'Yes'}
in_hospital_bowel_preparation_options = {0: 'No', 1: 'Yes'}
bowel_movement_status_options = {1: 'Normal', 2: 'Diarrhea', 3: 'Constipation'}
activity_level_options = {0: 'Frequently walking', 1: 'Occasionally walking'}
education_options = {1: 'Primary school or below', 2: 'Middle school', 3: 'High school or above'}

# Define feature names
feature_names = ['CCB', 'Last bowel movement was clear liquid', 'Split dose',
                 'In hospital bowel preparation', 'Bowel movement status', 'Activity level', 'Education']

# Streamlit user interface
st.title("Bowel Preparation Predictor")

# User inputs for prediction
ccb = st.selectbox("Use calcium channel blockers:", options=list(use_calcium_channel_blockers_options.keys()), format_func=lambda x: use_calcium_channel_blockers_options[x])
last_bowel_movement_was_clear_liquid = st.selectbox("Last bowel movement was clear liquid:", options=list(last_bowel_movement_was_clear_liquid_options.keys()), format_func=lambda x: last_bowel_movement_was_clear_liquid_options[x])
split_dose = st.selectbox("Split dose:", options=list(split_dose_options.keys()), format_func=lambda x: split_dose_options[x])
in_hospital_bowel_preparation = st.selectbox("In-hospital bowel preparation:", options=list(in_hospital_bowel_preparation_options.keys()), format_func=lambda x: in_hospital_bowel_preparation_options[x])
bowel_movement_status = st.selectbox("Bowel movement status:", options=list(bowel_movement_status_options.keys()), format_func=lambda x: bowel_movement_status_options[x])
activity_level = st.selectbox("Activity level:", options=list(activity_level_options.keys()), format_func=lambda x: activity_level_options[x])
education = st.selectbox("Education:", options=list(education_options.keys()), format_func=lambda x: education_options[x])

# Combine the user input features into a dataframe
feature_values = [ccb, last_bowel_movement_was_clear_liquid, split_dose,
                  in_hospital_bowel_preparation, bowel_movement_status, activity_level, education]
features = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Make predictions
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display the results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Display advice based on prediction
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, you have a high risk of poor bowel preparation. "
                  f"The model predicts that your probability of poor bowel preparation is {probability:.1f}%. "
                  "Consult your attending physician or charge nurse for better bowel preparation quality.")
    else:
        advice = (f"According to our model, you have a low risk of poor bowel preparation. "
                  f"The model predicts that your probability of good bowel preparation is {probability:.1f}%. "
                  "Follow medical instructions for bowel preparation.")

    st.write(advice)

# 创建 SHAP Explainer 对象，使用 predict_proba
explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_train, link="identity")

# 计算 SHAP 值，使用正类（失败类别）的 SHAP 值
shap_values = explainer.shap_values(custom_data)

# 确定正确的类别索引，例如类别 1 表示失败
fail_class_index = 1  # 假设类别 1 表示失败

 # 绘制局部解释 
    shap.initjs() 
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0], custom_data.iloc[0, :]) 
    file_name = "force_plot_" + str(time.time()) + ".html" 
    shap.save_html("./" + file_name, force_plot) 
    st.pyplot(plt.gcf())
