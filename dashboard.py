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
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    #     progress_text = "Operation in progress. Please wait."
    #     my_bar = st.progress(0, text=progress_text)

    #     my_bar.progress(percent_complete + 1, text=progress_text)

    

    

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


    #inputs = X.to_dict(orient="list")
    #prediction = requests.post(url=DEPLOYED_MODEL,
                            #json={"inputs": inputs},
                           #headers=headers)

    # Extract the response data as a dictionary
    #response_data = prediction.json()


    

    




    # Load the MLflow model
    model_uri = "mlruns/0/66bb88985bd041819b6f3f3bb9cf2d50/artifacts/mlflow_model"
    model = mlflow.pyfunc.load_model(model_uri)


   


    from lightgbm import LGBMClassifier

    
    # params = {
    # 'objective': 'binary',
    # 'metric': 'f1_score',
    # 'colsample_bytree': 0.4069597159452327,
    #  'learning_rate': 0.057736760620294536,
    #  'max_depth': 11, 
    #  'min_child_samples': 11,
    #  'n_estimators': 153,
    #  'num_leaves': 99,
    #  'reg_alpha': 9.822683433294355, 
    #  'reg_lambda': 5.167358912710143,
    #  'subsample': 0.3347462573473681,
    # 'random_state': 42
    # }
    # clf = LGBMClassifier(**params)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # # fit the classifier to the training data
    # clf.fit(X_train, y_train)


    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # import shap
    # shap_values = shap.TreeExplainer(clf).shap_values(X_test)
    # fig=shap.summary_plot(shap_values[1], X_test)

    # st.pyplot(fig)

    # Print the response data
    #st.write(response_data)
        #if submit:


    # progress_text = "Answered questions progress"
    # my_bar = st.progress(0, text=progress_text)

    # uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     bytes_data = uploaded_file.read()
    #     st.write("filename:", uploaded_file.name)
    #     st.write(bytes_data)
    
    # calc_contents = '''This is some text'''
    # st.download_button('Calculate',calc_contents)

    
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