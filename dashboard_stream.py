import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
from sklearn.model_selection import train_test_split
import pickle
import mlflow.pyfunc
import shap
from shap import maskers

#load data
df =pd.read_csv('cleaned.csv').drop(columns='Unnamed: 0')

categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
df=df.drop(columns=categorical_columns)
df=df.dropna()
#st.dataframe(df.head())
y=df['TARGET']
X = df.drop(columns=['TARGET','index'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


#load where model is deployed
DEPLOYED_MODEL = "http://127.0.0.1:5003/invocations"
headers = {"Content-Type": "application/json"}


#Main page interface
st.title('Home Credit APP')

   
    
    # data = pd.melt(chart_data.reset_index(), id_vars=["index"])

    # # Horizontal stacked bar chart
    # chart = (
    #     alt.Chart(data)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("value", type="quantitative", title=""),
    #         y=alt.Y("index", type="nominal", title=""),
    #         color=alt.Color("variable", type="nominal", title=""),
    #         order=alt.Order("variable", sort="descending"),
    #     )
    # )

    # st.altair_chart(chart, use_container_width=True)   #vertical version of the chart

col1, col2 = st.columns(([20,7]))

    
with col1:
    
    st.markdown('This is a tool to estimate clients capacity to repay a loan.')

    

    option = st.selectbox(
    'Choose a client?',
    ('Client1', 'Client2', 'Client3','Client4','Client5'))

    if option=='Client2':
        inputs = X.iloc[[1]].to_dict(orient="list")
        prediction = requests.post(url=DEPLOYED_MODEL,json={"inputs": inputs},headers=headers)
        response_data = prediction.json()
        response=response_data.get('predictions')[0]
        st.write(response)

    else:
        
        pass

    
    st.write("Results: ")
    text_contents = '''This is some text'''
    st.download_button('Download results',text_contents)
    st.write("Show statistics: ")





    # Load the MLflow model
    model_uri = "mlruns/0/66bb88985bd041819b6f3f3bb9cf2d50/artifacts/mlflow_model"
    model = mlflow.pyfunc.load_model(model_uri)


    
with col2:
    

    st.metric(label="Credit rate, %", value=4, delta=-0.5,
        delta_color="inverse")

    st.metric(label="Active credits", value=123456, delta=123,
        delta_color="off")
    
     # Load the SHAP values
    with open("shap_values.pkl", "rb") as f:
        shap_values = pickle.load(f)

 
    st.set_option('deprecation.showPyplotGlobalUse', False)

    fig=shap.summary_plot(shap_values[1], X_test)

    st.pyplot(fig)