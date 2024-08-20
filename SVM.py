{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2193566f-74a1-4e15-8651-6efda325b2c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load the model\n",
    "model = joblib.load('SVM.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "72ba94d2-9973-42ae-bbc0-c6da4e189474",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-08-20 10:41:10.696 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run F:\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-08-20 10:41:10.697 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Define feature options\n",
    "use_calcium_channel_blockers_options = {\n",
    "    0: 'No',\n",
    "    1: 'Yes'\n",
    "}\n",
    "\n",
    "last_bowel_movement_was_clear_liquid_options = {\n",
    "    0: 'No',\n",
    "    1: 'Yes'\n",
    "}\n",
    "\n",
    "split_dose_options = {\n",
    "    0: 'No',\n",
    "    1: 'Yes'\n",
    "}\n",
    "\n",
    "location_of_bowel_preparation_options = {\n",
    "    0: 'At home',\n",
    "    1: 'In hospital'\n",
    "}\n",
    "\n",
    "bowel_movement_status_options = {\n",
    "    1: 'Normal',\n",
    "    2: 'Diarrhea',\n",
    "    3: 'Constipation'\n",
    "}\n",
    "\n",
    "activity_level_options = {\n",
    "    0: 'Frequently walking',\n",
    "    1: 'Frequently walking'\n",
    "}\n",
    "\n",
    "education_options = {\n",
    "    1: 'Primary school or below',\n",
    "    2: 'Middle school',\n",
    "    3: 'High school or above'\n",
    "}\n",
    "\n",
    "# Define feature names\n",
    "feature_names = [\n",
    "    'CCB', 'Last bowel movement was clear liquid', 'Split-dose',\n",
    "    'In-hospital bowel preparation', 'Bowel movement status', 'Activity level', 'Education'\n",
    "]\n",
    "\n",
    "# Streamlit user interface\n",
    "st.title(\"Bowel Preparation Predictor\")\n",
    "\n",
    "# CCB: categorical selection\n",
    "ccb = st.selectbox(\"Use calcium channel blockers:\", options=list(use_calcium_channel_blockers_options.keys()), format_func=lambda x: use_calcium_channel_blockers_options[x])\n",
    "\n",
    "# Last bowel movement was clear liquid: categorical selection\n",
    "last_bowel_movement_was_clear_liquid = st.selectbox(\"Last bowel movement was clear liquid:\", options=list(last_bowel_movement_was_clear_liquid_options.keys()), format_func=lambda x: last_bowel_movement_was_clear_liquid_options[x])\n",
    "\n",
    "# Split-dose: categorical selection\n",
    "split_dose = st.selectbox(\"Split-dose:\", options=list(split_dose_options.keys()), format_func=lambda x: split_dose_options[x])\n",
    "\n",
    "# In-hospital bowel preparation: categorical selection\n",
    "location_of_bowel_preparation = st.selectbox(\"Location of bowel preparation:\", options=list(location_of_bowel_preparation_options.keys()), format_func=lambda x: location_of_bowel_preparation_options[x])\n",
    "\n",
    "# Bowel movement status: categorical selection\n",
    "bowel_movement_status = st.selectbox(\"Bowel movement status:\", options=list(bowel_movement_status_options.keys()), format_func=lambda x: bowel_movement_status_options[x])\n",
    "\n",
    "# Activity level: categorical selection\n",
    "activity_level = st.selectbox(\"Activity level:\", options=list(activity_level_options.keys()), format_func=lambda x: activity_level_options[x])\n",
    "\n",
    "# Education: categorical selection\n",
    "education = st.selectbox(\"Education:\", options=list(education_options.keys()), format_func=lambda x: education_options[x])\n",
    "\n",
    "# Process inputs and make predictions\n",
    "feature_values = ['CCB', 'Last bowel movement was clear liquid', 'Split-dose',\n",
    "                 'In-hospital bowel preparation', 'Bowel movement status', 'Activity level', 'Education']\n",
    "features = np.array([feature_values])\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    # Predict class and probabilities\n",
    "    predicted_class = model.predict(features)[0]\n",
    "    predicted_proba = model.predict_proba(features)[0]\n",
    "\n",
    "    # Display prediction results\n",
    "    st.write(f\"**Predicted Class:** {predicted_class}\")\n",
    "    st.write(f\"**Prediction Probabilities:** {predicted_proba}\")\n",
    "\n",
    "    # Generate advice based on prediction results\n",
    "    probability = predicted_proba[predicted_class] * 100\n",
    "\n",
    "    if predicted_class == 1:\n",
    "        advice = (\n",
    "            f\"According to our model, you have a high risk of poor bowl preparation. \"\n",
    "            f\"The model predicts that your probability of poor bowl preparation is {probability:.1f}%. \"\n",
    "            \"While this is just an estimate, it suggests that you may be at significant risk. \"\n",
    "            \"I recommend that you consult your attending physician or charge nurse \"\n",
    "            \"to achieve better bowel preparation quality.\"\n",
    "        )\n",
    "    else:\n",
    "        advice = (\n",
    "            f\"According to our model, you have a low risk of poor bowl preparation. \"\n",
    "            f\"The model predicts that your probability of good bowl preparation is {probability:.1f}%. \"\n",
    "            \"However, it is still very important to follow medical instructions for bowel preparation. \"\n",
    "            \"I recommend that you follow the advice of your doctor or charge nurse for bowel preparation. \"\n",
    "        )\n",
    "\n",
    "    st.write(advice)\n",
    "\n",
    "    # Calculate SHAP values and display force plot\n",
    "    explainer = shap.TreeExplainer(model)\n",
    "    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))\n",
    "\n",
    "    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)\n",
    "    plt.savefig(\"shap_force_plot.png\", bbox_inches='tight', dpi=1200)\n",
    "\n",
    "    st.image(\"shap_force_plot.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba77aad9-5840-4201-9a7e-bac817157207",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
