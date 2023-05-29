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
csv_url = 'https://raw.githubusercontent.com/olga-sonina/for_a/a98e78a19b3af564fb92823d61366d6c86e079b2/cleaned1000.csv'


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




#SHAP
st.subheader(
    "SHAP values for a particular client"

    )
#Client section

lst = list(range(1,len(y)-1))
list_string = map(str, lst)
client_list = ['Client'+ x  for x in list_string]
option = st.selectbox('Choose a client',client_list)
n_client=client_list.index(option)
agree = st.checkbox("Show details")
if agree:
    st.write(X.iloc[[n_client]])
# Plot Waterfall Plot
st.write('SHAP values: ')
expected_value = shap_values[0].mean(axis=1)
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = shap.force_plot(expected_value[n_client], shap_values[0][n_client], X.iloc[n_client,:], matplotlib=True)

# Display the plot

st.pyplot(fig)




st.subheader('SHAP values of LGBM classifier model for the entire dataset')
st.set_option('deprecation.showPyplotGlobalUse', False)
fig1=shap.summary_plot(shap_values[0], X,cmap = "plasma")
st.pyplot(fig1)

# Show histogram of TARGET
#fig, ax = plt.subplots()
#ax.hist(df['TARGET'], bins=20)
#st.pyplot(fig)

#Feature importances



# def allowSelfSignedHttps(allowed):
#     # bypass the server certificate verification on client side
#     if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
#         ssl._create_default_https_context = ssl._create_unverified_context

# allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# def calc_client(option):
#     n_client=client_list.index(option)
#     inputs=X.iloc[[n_client]].to_json(orient="split")
#     # Request data goes here
#     # The example below assumes JSON formatting which may be updated
#     # depending on the format your endpoint expects.
#     # More information can be found here:
#     # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

#     data = ast.literal_eval(inputs)
#     body = str.encode(json.dumps(data))
#     url = "http://d2b1f5a8-29cc-4b8d-8a28-165d9f2404d9.francecentral.azurecontainer.io/score"
#     headers = {'Content-Type':'application/json'}
#     req = urllib.request.Request(url, body, headers)

#     try:
#         response = urllib.request.urlopen(req)
#         result = response.read()
#         st.write(result.decode("utf-8") )
#     except urllib.error.HTTPError as error:
#         st.write("The request failed with status code: " + str(error.code))
#         # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
#         st.write(error.info())
#         st.write(error.read().decode("utf8", 'ignore'))

# #calc_client(option)

