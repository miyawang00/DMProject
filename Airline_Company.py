# Import the required packages
# test
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

# Title
st.title("Airline Company Analysis")


def load_data():
    # Load data
    df = pd.read_csv('flight data.csv', on_bad_lines='skip')
    df = df.drop_duplicates()
    df = df.dropna()

    # Clean the flight_number feature
    df['flight_number'] = df['flight_number'].str.split('|').str[0]

    # Clean the airline_name feature
    df['airline_name'] = df['airline_name'].str.replace('[', '').str.replace(']', '')
    df['airline_name'] = df['airline_name'].str.split('|').str[0]

    return df


df = load_data()


with st.expander("Show the 'Flights' dataframe"):
    st.write(df)

# CO2 / Price Analysis

st.header("CO2 / Price Analysis among Airline Companies")

# Clean the co2_percentage feature
df['co2_percentage'] = df['co2_percentage'].str.replace('%', '')
# Get the diff between co2_emissions and avg_co2_emissions_for_this_route
df['co2'] = df['co2_emissions'] - df['avg_co2_emission_for_this_route']
# Get the price per duration
df['price_by_duration'] = df['price'] / df['duration']

metric = df[['co2', 'price_by_duration']]
new_slider_01 = [col for col in metric]

cole, col1, cole, col2, cole = st.columns([0.1, 1, 0.05, 1, 0.1])

with col1:
    MetricSlider01 = st.selectbox("Pick your metric:", new_slider_01)

with col2:
    company_list = sorted(df['airline_name'].drop_duplicates().tolist())
    multiselect = st.multiselect("Select the airline company:", company_list, ['LATAM', 'Air France'])
    st.write("You selected", len(multiselect), 'airline companies')
    transaction_df = df[df['airline_name'].isin(multiselect)]

try:
    if MetricSlider01 == 'co2':
        co2_by_airline = transaction_df.groupby("airline_name")['co2'].mean().reset_index()
        sorted_co2 = co2_by_airline.sort_values("co2", ascending = False)

        fig = px.bar(sorted_co2, x = "airline_name", y = "co2", color = "airline_name",
                     labels = {"airline_name" : "Airline Company", "co2" : "Average CO2 Emissions (KG) Above The Standard Emissions"},
                     title = "Airline Companies With The Highest Average CO2 Emissions Above The Standard Emissions")

        st.plotly_chart(fig)

    elif MetricSlider01 == 'price_by_duration':
        price_by_airline = transaction_df.groupby("airline_name")['price_by_duration'].mean().reset_index()
        sorted_price = price_by_airline.sort_values("price_by_duration", ascending=False)

        fig = px.bar(sorted_price, x="airline_name", y="price_by_duration", color="airline_name",
                     labels={"airline_name": "Airline Company",
                             "price_by_duration": "Price (USD) Per Duration Time"},
                     title="Airline Companies With The Highest Average Price By Duration Time")

        st.plotly_chart(fig)

except IndexError:
    st.warning("Throwing an exception!")


# Stop - Price Analysis

st.header("Stops And Ticket Price Relationship Analysis")

company_list = sorted(df['airline_name'].drop_duplicates().tolist())
MetricSlider03 = st.selectbox("Select the airline company that you want to check:", company_list)
transaction_df_2 = df[df['airline_name'] == MetricSlider03]

try:
    fig_stop = px.box(transaction_df_2, x = "stops", y = "price", color = "stops",
                      labels = {"stops": "Number of Stops",
                                "price": "Ticket Price (USD)"},
                      title = f"Relationship Between Number Of Stops Of {MetricSlider03} Airline Company")

    st.plotly_chart(fig_stop)

except IndexError:
    st.warning("Throwing an exception!")
    
### Changes By Nancy
