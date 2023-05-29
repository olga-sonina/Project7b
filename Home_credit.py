import streamlit as st
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
import warnings
warnings.filterwarnings("ignore")

st.write("# Home credit App")


st.markdown(
    """
    A tool predicting client's capacity to repay a loan."""

    )# GitHub raw CSV file URL
csv_url ="https://raw.githubusercontent.com/olga-sonina/for_a/a98e78a19b3af564fb92823d61366d6c86e079b2/cleaned1000.csv"
#https://github.com/olga-sonina/Project7/blob/cd489dd49856aee31b41230781f1c404dca7e60f/cleaned1000.csv

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

#Client section
lst = list(range(1,len(y)-1))
list_string = map(str, lst)
client_list = ['Client'+ x  for x in list_string]
option = st.selectbox('Choose a client',client_list)
n_client=client_list.index(option)




def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

def calc_client(option):
    n_client=client_list.index(option)
    inputs=X.iloc[[n_client]].to_json(orient="split")
    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

    data = ast.literal_eval(inputs)
    body = str.encode(json.dumps(data))
    url = "http://d2b1f5a8-29cc-4b8d-8a28-165d9f2404d9.francecentral.azurecontainer.io/score"
    headers = {'Content-Type':'application/json'}
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        st.write(result.decode("utf-8") )
    except urllib.error.HTTPError as error:
        st.write("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        st.write(error.info())
        st.write(error.read().decode("utf8", 'ignore'))

#calc_client(option)

if st.button('Predict'):
	#calc_client(option)
    #st.write(option)

#add checkbox
agree = st.checkbox("Show details")

if agree:
    st.write(X.iloc[[n_client]])
    # Plot Waterfall Plot
    st.write('SHAP values: ')
    try:
        with open("shap_values.pkl", "rb") as f:
            shap_values = pickle.load(f)
    except Exception as e:
        st.write(f"Error loading SHAP values: {e}")
    expected_value = shap_values[0].mean(axis=1)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.force_plot(expected_value[n_client], shap_values[0][n_client], X.iloc[n_client,:], matplotlib=True)
    # Display the plot
    st.pyplot(fig)
