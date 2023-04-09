import pandas as pd
import streamlit as st
import joblib
st.write("Flight Price")

col1, col2 = st.columns(2)
from_country = col1.selectbox("Enter your from country",['Algeria'])
dest_country = col2.selectbox("Enter your destination country", ['Argentina', 'Australia', 'Austria', 'Brazil', 'Belgium', 'Canada',
       'Chile', 'China', 'Columbia', 'Denmark', 'Dublin', 'Ethiopia',
       'Egypt', 'France', 'Germany', 'India', 'Greece', 'Indonesia',
       'Italy', 'Japan', 'Kenya', 'Mexico', 'Malaysia', 'Morocco',
       'Norway', 'Netherlands', 'Panama', 'Philippines', 'Peru',
       'Portugal', 'Rome', 'Qatar', 'Russia', 'Singapore', 'South Africa',
       'South Korea', 'Spain', 'Taiwan', 'Sweden', 'Thailand'])
departure_time = col1.selectbox("Enter your departure time",['2022-04-30', '2022-05-02', '2022-05-06', '2022-05-14',
       '2022-05-29', '2022-07-28', '2022-08-27', '2022-05-03',
       '2022-05-15', '2022-08-28', '2022-05-07', '2022-05-30',
       '2022-07-29', '2022-05-01'])
arrival_time = col2.selectbox("Entern your arrival time", ['2022-05-01', '2022-05-02', '2022-05-04', '2022-05-03',
       '2022-05-05', '2022-05-08', '2022-05-09', '2022-05-07',
       '2022-05-16', '2022-05-17', '2022-05-15', '2022-05-31',
       '2022-06-01', '2022-05-30', '2022-07-29', '2022-07-30',
       '2022-08-28', '2022-08-29', '2022-04-30', '2022-05-06',
       '2022-05-14', '2022-05-29', '2022-07-28', '2022-08-27',
       '2022-07-31', '2022-08-01', '2022-08-31', '2022-08-30',
       '2022-06-02', '2022-09-01', '2022-05-18', '2022-05-11',
       '2022-05-10', '2022-05-19', '2022-05-12', '2022-06-03',
       '2022-08-02', '2022-09-02'] )
st.button("Predict")
                              
