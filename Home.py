import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Plotting Data", page_icon="📈")
st.header('Plotting Data')
option_list = sorted(os.listdir('dataset'))
dataset_selected = st.selectbox(label='Select dataset', options=option_list)

col1, col2 = st.columns(2)
def PlotDataset(dataset):
  file_original = f'dataset/{dataset}'
  df_original = pd.read_csv(file_original)
  
  file_resampled_train = f'dataset_resampled/train/{dataset}'
  df_resampled_train = pd.read_csv(file_resampled_train)
  file_resampled_test = f'dataset_resampled/test/{dataset}'
  df_resampled_test = pd.read_csv(file_resampled_test)
  df_resampled = pd.concat([df_resampled_train, df_resampled_test])
  
  with col1:
    st.subheader('Original Data')
    st.write(df_original.head())
    st.write('Number of records:', df_original.shape[0])
    
    st.subheader('Fraud/Non-fraud distribution')
    fig1 = px.pie(df_original, names='fraud_bool')
    st.plotly_chart(fig1)
    #----------
    st.subheader('Age distribution')
    fig2 = px.histogram(df_original, x='customer_age', color='fraud_bool')
    st.plotly_chart(fig2)
    #----------
    st.subheader('Transaction amount distribution') 
    fig3 = px.box(df_original, x='fraud_bool', y='intended_balcon_amount')
    st.plotly_chart(fig3)
    #----------
    st.subheader('Correlation matrix')
    numeric_cols = df_original.select_dtypes(include=['float64', 'int64']).columns
    corr = df_original[numeric_cols].corr()
    fig4 = px.imshow(corr)
    st.plotly_chart(fig4)
    
   
  with col2:
    st.subheader('Resampled Data')
    st.write(df_resampled.head())
    st.write('Number of records:', df_resampled.shape[0])
    
    st.subheader('Fraud/Non-fraud distribution')
    fig5 = px.pie(df_resampled, names='fraud_bool')
    st.plotly_chart(fig5)
    #----------
    st.subheader('Age distribution')
    fig6 = px.histogram(df_resampled, x='customer_age', color='fraud_bool')
    st.plotly_chart(fig6)
    #----------
    st.subheader('Transaction amount distribution') 
    fig7 = px.box(df_resampled, x='fraud_bool', y='intended_balcon_amount')
    st.plotly_chart(fig7)
    #----------
    st.subheader('Correlation matrix')
    numeric_cols = df_resampled.select_dtypes(include=['float64', 'int64']).columns
    corr = df_resampled[numeric_cols].corr()
    fig8 = px.imshow(corr)
    st.plotly_chart(fig8)
  
PlotDataset(dataset_selected)
