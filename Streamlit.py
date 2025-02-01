import numpy as np
import streamlit as st
import pickle
import pandas as pd


with open('Passenger.pkl', 'rb') as file:
    model = pickle.load(file)
    
st.title('Passenger survival App')
st.subheader("survival")

uploaded_file = st.file_uploader("Choose a csv file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    prediction = model.predict(data)
    st.write(prediction)

