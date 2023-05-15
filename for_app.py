import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests


#load data
#df =pd.read_csv('cleaned.csv').drop(columns='Unnamed: 0')


#Main page interface
st.title('Home Credit App')

   
    
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

    client_list=['Client1', 'Client2', 'Client3','Client4','Client5', 'Client6']

    option = st.selectbox('Choose a client',client_list)
    st.write('you chose: ', option)

    # def calc_client(option):
    #     n_client=client_list.index(option)
    #     inputs = X.iloc[[n_client]].to_dict(orient="list")
    #     prediction = requests.post(url=DEPLOYED_MODEL,json={"inputs": inputs},headers=headers)
    #     response_data = prediction.json()
    #     response=response_data.get('predictions')[0]
    #     st.write(response)

    # calc_client(option)

    
    # st.write("Results: ")
    # text_contents = '''This is some text'''
    # st.download_button('Download results',text_contents)
    # st.write("Show statistics: ")





    # # Load the MLflow model
    # model_uri = "mlruns/0/66bb88985bd041819b6f3f3bb9cf2d50/artifacts/mlflow_model"
    # model = mlflow.pyfunc.load_model(model_uri)


    
with col2:

    st.metric(label="Credit rate, %", value=4, delta=-0.5,
        delta_color="inverse")

    st.metric(label="Active credits", value=123456, delta=123,
        delta_color="off")
    
     # Load the SHAP values
    # with open("shap_values.pkl", "rb") as f:
    #     shap_values = pickle.load(f)

 
    # st.set_option('deprecation.showPyplotGlobalUse', False)

    # fig=shap.summary_plot(shap_values[1], X_test)

    # st.pyplot(fig)