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



# GitHub raw CSV file URL
csv_url ='https://raw.githubusercontent.com/olga-sonina/for_a/a98e78a19b3af564fb92823d61366d6c86e079b2/cleaned1000.csv'
# Function to fetch and load CSV data
cl_csv="https://raw.githubusercontent.com/olga-sonina/for_a/2d1fb07371af7d3ce791f31d909a6640ae82a015/clients.csv"
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
df = df.drop(columns="Unnamed: 0")# Drop the 'Unnamed: 0' column
df_c=load_data(cl_csv).drop(columns="Unnamed: 0").iloc[:,0].values

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

group_0=find_indices(df_c, lambda e: e <0.6)
group_1=find_indices(df_c, lambda e: (e >0.6) & (e<0.83))
group_2=find_indices(df_c, lambda e: (e >0.83) & (e<0.99))
group_3=find_indices(df_c, lambda e: e>0.99)
# group_0_mean=df.iloc[group_0].mean(axis=0)
# group_1_mean=df.iloc[group_1].mean(axis=0)
# group_2_mean=df.iloc[group_2].mean(axis=0)
# group_3_mean=df.iloc[group_3].mean(axis=0)
# st.write(df.iloc[group_0].mean(axis=0))
# Display the DataFrame in Streamlit app
#if df is not None:
    #st.dataframe(df.head())
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
df=df.drop(columns=categorical_columns)
df=df.dropna()
#st.dataframe(df.head())
y=df['TARGET'].astype('float')
X = df.drop(columns=['TARGET','index']).astype('float')

with st.container():
    col1,col2=st.columns(2)
    with col1:
        #Dataset
        st.subheader('Dataset general information: ')
        st.write('Number of clients: 307511')
        st.write('Number of parameters: 163')
        st.write('Missing values rate after cleansing: 0.24395941907129431')
        st.write('Duplicated lines: 0')

        # # Show histogram of TARGET
        # fig, ax = plt.subplots()
        # ax.hist(df['TARGET'], bins=20)
        # st.pyplot(fig)

    with col2:
        # Count the frequency of each unique value in the 'TARGET' column
        value_counts = df['TARGET'].value_counts()

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        ax.set_aspect('equal')  # Ensure the pie chart is circular
        ax.set_title('Distribution of TARGET')
        plt.show()
        st.pyplot(fig)

st.subheader('Explore distribution of variables and their correlations')
#Construct graph
param_list=X.columns.tolist()
param_opt_x=st.selectbox('Choose parameter x',param_list, index=5)
param_opt_y=st.selectbox('Choose parameter y',param_list, index=11)
fig = px.scatter(
    data_frame=df,
    x=df[param_opt_x],
    y=df[param_opt_y],
    color=df["TARGET"]
    #title=""
)
st.plotly_chart(fig)





#Client section

lst = list(range(1,len(y)-1))
list_string = map(str, lst)
client_list = ['Client'+ x  for x in list_string]
option = st.selectbox('Choose a client',client_list)
n_client=client_list.index(option)
agree = st.checkbox("Show details")



if agree:
    st.write(X.iloc[[n_client]])


if n_client in group_0:
    X_group=df.iloc[group_0]#.mean(axis=0)
    st.write("Client's group: ",'Not allowed')
if n_client in group_1:
    X_group=df.iloc[group_1]#.mean(axis=0)
    st.write("Client's group: ", "Potentially could be approved with improved parameters")
if n_client in group_2:
    X_group=df.iloc[group_2]#.mean(axis=0)
    st.write("Client's group: ","Likely approved")
if n_client in group_3:
    X_group=df.iloc[group_3]#.mean(axis=0)
    st.write("Client's group: ","Approved, no-risk clients")
    
    

mean_client=X_group.mean(axis=0)

chart_data = pd.DataFrame(
    mean_client,
    columns=['mean_client_in_the group'])
new_client=X.iloc[n_client,:]
chart_data['Client']=new_client

st.line_chart(chart_data)


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
