import streamlit as st
import xgboost as xgb
import lightgbm as lgb
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo", page_icon="📊")
st.header('Demo')
dataset_options = sorted(os.listdir('/Users/cnmeow/streamlit/dataset_resampled/test'))
dataset_selected = st.selectbox(label='Select dataset', options=dataset_options)
name_dataset = dataset_selected.split('.')[0]
xgb_options = sorted(os.listdir(f'/Users/cnmeow/streamlit/checkpoints/xgboost'))
lgb_options = sorted(os.listdir('/Users/cnmeow/streamlit/checkpoints/lightgbm'))

col1, col2 = st.columns(2)
with col1:
  xgb_selected = st.selectbox(label='Select checkpoint for XGBoost', options=xgb_options)

with col2:
  lgb_selected = st.selectbox(label='Select checkpoint for LightGBM', options=lgb_options)

def Demo(xgb_selected, lgb_selected, dataset_selected):
  test_file = f'/Users/cnmeow/streamlit/dataset_resampled/test/{dataset_selected}'
  df_test = pd.read_csv(test_file)
  X_test = df_test.drop(columns=['fraud_bool'])
  y_test = df_test['fraud_bool']
  
  xgboost_checkpoint = f'/Users/cnmeow/streamlit/checkpoints/xgboost/{xgb_selected}'
  xgboost_loaded = xgb.Booster(model_file=xgboost_checkpoint)
  xgboost_dtest = xgb.DMatrix(X_test, label=y_test)
  xgboost_pred = xgboost_loaded.predict(xgboost_dtest)
  xgboost_pred_binary = (xgboost_pred >= 0.5).astype(int)
  xgboost_accuracy = accuracy_score(y_test, xgboost_pred_binary)
  xgboost_classification_report = classification_report(y_test, xgboost_pred_binary)
  xgboost_roc_auc = roc_auc_score(y_test, xgboost_pred_binary)
  with col1:
    st.subheader('XGBoost')
    st.write("Accuracy:", xgboost_accuracy)
    st.write("Classification Report:")
    st.code(xgboost_classification_report, language="plaintext") 
    st.write("ROC AUC:", xgboost_roc_auc)

    # Tính toán FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, xgboost_pred_binary).ravel()
    fpr = fp / (fp + tn)  # Tính FPR
    st.write("False Positive Rate (FPR):", fpr)

    # Trực quan hóa feature importance
    st.subheader("Feature Importance")
    fig1, ax1 = plt.subplots(figsize=(10, 6)) 
    xgb.plot_importance(xgboost_loaded, importance_type='weight', max_num_features=10, height=0.5, ax=ax1)
    st.pyplot(fig1)
  
  lightgbm_checkpoint = f'/Users/cnmeow/streamlit/checkpoints/lightgbm/{lgb_selected}'
  lightgbm_loaded = lgb.Booster(model_file=lightgbm_checkpoint)
  lightgbm_pred = lightgbm_loaded.predict(X_test, num_iteration=lightgbm_loaded.best_iteration)
  lightgbm_pred_binary = (lightgbm_pred >= 0.5).astype(int)
  lightgbm_accuracy = accuracy_score(y_test, lightgbm_pred_binary)
  lightgbm_classification_report = classification_report(y_test, lightgbm_pred_binary)
  lightgbm_roc_auc = roc_auc_score(y_test, lightgbm_pred_binary)
  with col2:
    st.subheader('LightGBM')
    st.write("Accuracy:", lightgbm_accuracy)
    st.write("Classification Report:")
    st.code(lightgbm_classification_report, language="plaintext") 
    st.write("ROC AUC:", lightgbm_roc_auc)
    
    # Tính toán FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, lightgbm_pred_binary).ravel()
    fpr = fp / (fp + tn)  # Tính FPR
    st.write("False Positive Rate (FPR):", fpr)
  
    # Trực quan hóa feature importance
    st.subheader("Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(10, 6)) 
    lgb.plot_importance(lightgbm_loaded, max_num_features=10, importance_type='split', ax=ax2)
    st.pyplot(fig2)
    
Demo(xgb_selected, lgb_selected, dataset_selected)