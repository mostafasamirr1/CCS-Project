import streamlit as st
import pandas as pd


st.title(':rainbow[CCS Equations Site]')
st.header('Site Developed by: Mostafa Samir-Mahmoud Elseginy')
st.title(':rainbow[TEAM MEMBERS SUS]')

df = pd.read_csv("names.csv", index_col=0)
df['id'] = df['id'].astype(str)

st.data_editor(df)
st.snow()
st.balloons()