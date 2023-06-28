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

    )
csv_url =st.secrets.urls.df

# Function to fetch and load CSV data
@st.cache
def load_data(csv):
    try:
        # Fetch CSV data from URL
        return pd.read_csv(csv)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load CSV data
df = load_data(csv_url)
cl_csv="https://raw.githubusercontent.com/olga-sonina/for_a/2d1fb07371af7d3ce791f31d909a6640ae82a015/clients.csv"
df_c=load_data(cl_csv)
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
    #url = "http://d2b1f5a8-29cc-4b8d-8a28-165d9f2404d9.francecentral.azurecontainer.io/score"
    url="https://my-workspace-dep-bhebh.francecentral.inference.ml.azure.com/score"
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
resp=df_c.iloc[n_client].loc['0']#predicted probability to 0 class (approved loan class)
print("Score of client: ",resp)
threshold=0.83 #threshold found by cout-metier function
def client_group(resp, threshold):
    if resp <0.6:
        client_group="Not approved"
    if (resp >0.6) & (resp<threshold):
        client_group="Potentially could be approved with improved parameters"
    if (resp>threshold) & (resp<0.99):
        client_group="Likely approved"
    if resp>0.99:
        client_group="Approved, no-risk clients"
        
    return client_group

client_g=client_group(resp, threshold)

if st.button('Predict'):
   #calc_client(option)
   st.write("Score: ", resp)
   st.write("Current threshold: ", threshold)
   st.write("Client's group: ", client_g)
   if resp>threshold:
       st.write("Congrats, loan approved! ")
   else:
       st.write("Loan declined. Look at the client's statistics to identify parameters for improvement compared to other clients in their group")



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
    st.write("By examining the SHAP values, one can determine which features have the greatest influence on the prediction for the chosen client. Positive SHAP values (red) indicate features that increase the score, while negative values (blue) indicate features that decrease it.")
