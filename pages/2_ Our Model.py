import os
import ssl
import ast
import sys
import json
import requests
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



# GitHub raw CSV file URL
csv_url = st.secrets.urls.df


# Function to fetch and load CSV data
@st.cache
def load_data():
    try:
        # Fetch CSV data from URL
        return pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load CSV data
df = load_data()
# Drop the 'Unnamed: 0' column
df = df.drop(columns="Unnamed: 0")
# Display the DataFrame in Streamlit app
#if df is not None:
    #st.dataframe(df.head())
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
df=df.drop(columns=categorical_columns)
df=df.dropna()
#st.dataframe(df.head())
y=df['TARGET'].astype('float')
X = df.drop(columns=['TARGET','index']).astype('float')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
## Load the SHAP values
st.subheader(
    "Model"

    )
st.write('Our prediction is based on LGBM classifier model. LGBM uses a tree-based learning algorithm that constructs decision trees in a leaf-wise manner, which helps to achieve better accuracy with fewer iterations compared to other boosting algorithms. It is designed to handle large datasets and has gained popularity for its excellent performance in various machine learning tasks.  ')
st.subheader(
    "Feature Importances"

    )
st.write('Feature Importances helps us identify the most influential features in the model and gain insights into the underlying relationships between the features and the target variable. By analyzing feature importances, we can prioritize feature selection, identify potential feature interactions, and improve our understanding of the problem at hand.')
feat_imp=pd.read_csv('feat_imp.csv').drop(columns='Unnamed: 0')
st.dataframe(feat_imp.sort_values(by='importance', ascending=False)[0:10])
#plt.figure(figsize=(8, 10))
#fig=sns.barplot(x='importance',y='feats', data=feat_imp.sort_values(by='importance', ascending=False)[0:50])
#plt.title('LightGBM Features')
#plt.tight_layout()
#st.pyplot(fig)



try:
    with open("shap_values.pkl", "rb") as f:
        shap_values = pickle.load(f)
except Exception as e:
    st.write(f"Error loading SHAP values: {e}")




st.subheader('SHAP values of LGBM classifier model for the entire dataset')
st.write("By examining the SHAP values, one can determine which features have the greatest influence on the prediction.",
 "For example, if most of the data points have high positive SHAP values for the 'INCOME' feature, it suggests that higher income has a more significant contribution to increasing the predicted probability of loan approval.",
 " This implies that 'INCOME' plays a crucial role in determining the loan approval decision according to the model.")



st.set_option('deprecation.showPyplotGlobalUse', False)
fig1=shap.summary_plot(shap_values[0], X,cmap = "plasma")
st.pyplot(fig1)
