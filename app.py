import streamlit as st
import joblib
import pandas as pd

# load model
price_model= joblib.load("/Users/gohzhengyong/Desktop/Machine_Learning_Projects/Project1/model/price_model.pkl")
region_model= joblib.load("/Users/gohzhengyong/Desktop/Machine_Learning_Projects/Project1/model/region_model.pkl")

# slidebar input
st.sidebar.header("Enter user information")
location = st.sidebar.selectbox("Location", ['KL', 'Selangor', 'PJ', 'Kajang', 'Sunway'])
num_bath = st.sidebar.slider("Number of bathrooms", 1,10)
