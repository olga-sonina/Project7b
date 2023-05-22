import urllib.request
import json
import os
import ssl
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import ast
import os
import sys
from pathlib import Path

import setuptools
from setuptools.command.install import install

THIS_DIRECTORY = Path(__file__).parent

VERSION = "1.22.0"  # PEP-440

NAME = "streamlit"

# IMPORTANT: We should try very hard *not* to add dependencies to Streamlit.
# And if you do add one, make the required version as general as possible:
# - Include relevant lower bound for any features we use from our dependencies
# - And include an upper bound that's < NEXT_MAJOR_VERSION
INSTALL_REQUIRES = [
"plotly",
"plotly.express"]


#import matplotlib.pyplot as plt


#Main page interface
st.title('Home Credit App')
import plotly
import plotly.express as px

# GitHub raw CSV file URL
csv_url = 'https://raw.githubusercontent.com/olga-sonina/Project7/c4697ffb89de47c603d0165a67270c2710a75060/cleaned1000.csv'

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


#fig, ax = plt.subplots()
#ax.hist(df['TARGET'], bins=20)

#st.pyplot(fig)
#json_data = json.dumps(json_data)
#df = pd.read_json(json_data)
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
df=df.drop(columns=categorical_columns)
df=df.dropna()
#st.dataframe(df.head())
y=df['TARGET'].astype('float')
X = df.drop(columns=['TARGET','index']).astype('float')
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
param_list=X.columns.tolist()

param_opt_x=st.selectbox('Choose a parameter x',param_list)
param_opt_y=st.selectbox('Choose a parameter y',param_list)
fig = px.scatter(
    data_frame=df,
    x=df[param_opt_x],
    y=df[param_opt_y],
    color=df["TARGET"]
    #title="Records to Highlight"
)
st.plotly_chart(fig)


#Client section

lst = list(range(1,len(y)-1))
list_string = map(str, lst)
client_list = ['Client'+ x  for x in list_string]
option = st.selectbox('Choose a client',client_list)
n_client=client_list.index(option)
st.write(X.iloc[[n_client]])


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

