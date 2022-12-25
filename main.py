# Libraries
import streamlit as st
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from babel.numbers import format_decimal
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
import numpy as np


# Page Config 
st.set_page_config(page_title='Dash', page_icon='👽')

# Branding
st.title("ML Modeling with Sam 👽")
st.text('''
This is a Data Science Project on House Price Prediction 
using the Multi-Variate Linear Regression Algorithm. Inshort
the algorithm takes some data as input and predicts the price. ''')

st.write('##')
st.write('##')

# Dataset Part
def get_main_data(filename):
    df=pd.read_csv(filename)
    return df
    
def get_sub_data(filename):
    sdf=pd.read_csv(filename)
    return sdf

# Page Content

    # Sidebar

st.sidebar.image('logo.png',width=310)
st.sidebar.title('Content')
st.sidebar.write(" 🚀Intro [➤](#ml-modeling-with-sam)")
st.sidebar.write(" 📉Trends [➤](#houseprice-trends)")
st.sidebar.write(" 💡Features[➤](#features)")
st.sidebar.write(" 🛸Algorithm[➤](#about-the-algorithm)")
st.sidebar.write(" 🛸Prediction Model[➤](#model)")


# Data Access and Optimization
sdf = get_sub_data('sub.csv')
df=get_main_data('mumbai.csv')
price_dist = pd.DataFrame(df['price'].tail(1000))
st.subheader('Houseprice Trends')
st.bar_chart(price_dist)

# Feature 
st.title('Features')
st.text('''
So this data set is collected for completing a academic project ,which is a web
app for calculating the price of  houses. This data  is  scraped  from  magic bricks 
website between june  2021 and july  2021. With the  help of the data available one 
can make a regression model to predict house prices.
''')
st.text('Area')
st.text('Bedrooms')
st.text('Bathrooms')
st.text('Balconies')

 
st.title('About The Algorithm')
st.header("Multi Variate Regression Model")
st.image('mlvr.png',width=700,caption='*For Explainatory purposes only')
st.write('##')
st.text('''
Most of the statistically analysed data does not necessarily have one response va-
riable and one explanatory variable. In  most cases, the number  of  variables can
vary depending on the study.To measure the relationships between the multidimensi-
onal variables, multivariate regression is used.

Multivariate regression is a technique used to measure   the degree to   which the
various independent variable and various dependent variables are linearly  related
to each other.The relation is said to be linear due to the correlation between the
variables. Once the multivariate regression is applied to the dataset, this method
is then used to predict the behaviour of the response variable based on its corres-
ponding predictor variables. 
''')


# Model Container
model_training = st.container()

with model_training:
    st.title('Model')
    st.text('Model Trained With Multi-Variate Linear Regression.')
    in_put,out_put = st.columns(2)
    area=in_put.slider('Area in sqft',min_value=0,max_value=6000,value=500,step=10)
    bedrooms=in_put.slider('No of Bedroom(s)',min_value=0,max_value=10,value=3,step=1)
    bathrooms=in_put.slider('No of Bathroom(s)',min_value=1,max_value=6,value=2,step=1)
    parking=in_put.slider('Parking Space',min_value=0,max_value=3,value=1,step=1)
    lift=in_put.slider('No of Lifts',min_value=0,max_value=5,value=1,step=1)
    ok =st.button('Calculate Price')

    # Ml Model
    reg = linear_model.LinearRegression()
    reg.fit(df[['area','Bedrooms','Bathrooms','parking','Lift']],df.price)
    prediction=reg.predict([[area,bedrooms,bathrooms,parking,lift]])
    out_put.text('The Price that the Model Predicted was')
    if ok == True:
        out_put.title(f"₹{int(prediction)}")

st.write("##")
st.write("##")
st.write("##")
# Dataset Preview

st.write("Dataset Preview")
checkbox = st.checkbox("View")

if checkbox==True:
    st.dataframe(sdf)





