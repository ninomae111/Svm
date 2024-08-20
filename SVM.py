import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# 加载SVM模型
model = joblib.load('SVMNEW.pkl')

# 定义特征选项
use_calcium_channel_blockers_options = {
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

# 定义特征名称
feature_names = [
    'CCB', 
    'Last bowel movement was clear liquid', 
    'Split dose',
    'In hospital bowel preparation', 
    'Bowel movement status', 
    'Activity level', 
    'Education'
]

# Streamlit用户界面
st.title("Bowel Preparation Predictor")

# 特征选择
ccb = st.selectbox("Use calcium channel blockers:", options=list(use_calcium_channel_blockers_options.keys()), format_func=lambda x: use_calcium_channel_blockers_options[x])
last_bowel_movement_was_clear_liquid = st.selectbox("Last bowel movement was clear liquid:", options=list(last_bowel_movement_was_clear_liquid_options.keys()), format_func=lambda x: last_bowel_movement_was_clear_liquid_options[x])
split_dose = st.selectbox("Split dose:", options=list(split_dose_options.keys()), format_func=lambda x: split_dose_options[x])
in_hospital_bowel_preparation = st.selectbox("In hospital bowel preparation:", options=list(in_hospital_bowel_preparation_options.keys()), format_func=lambda x: in_hospital_bowel_preparation_options[x])
bowel_movement_status = st.selectbox("Bowel movement status:", options=list(bowel_movement_status_options.keys()), format_func=lambda x: bowel_movement_status_options[x])
activity_level = st.selectbox("Activity level:", options=list(activity_level_options.keys()), format_func=lambda x: activity_level_options[x])
education = st.selectbox("Education:", options=list(education_options.keys()), format_func=lambda x: education_options[x])

# 将输入数据转换为NumPy数组
input_data = np.array([
    ccb,
    last_bowel_movement_was_clear_liquid,
    split_dose,
    in_hospital_bowel_preparation,
    bowel_movement_status,
    activity_level,
    education
]).reshape(1, -1)

# 显示用户输入的数据
st.write("输入的数据为:")
st.write(dict(zip(feature_names, input_data.flatten())))

# 计算SHAP值
explainer = shap.KernelExplainer(model.predict_proba, input_data)
shap_values = explainer.shap_values(input_data)

# 绘制SHAP力图
st.subheader("SHAP力图展示")
shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True)

# 将图像显示在Streamlit中
st.pyplot(bbox_inches='tight')
