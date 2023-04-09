import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

df = pd.read_csv('flight data.csv', on_bad_lines='skip')
# Drop duplicate values
df.drop_duplicates(inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# remove the airline_name column in  pandas DataFrame that contains square brackets []

df['airline_name'] = df['airline_name'].str.replace('[','').str.replace(']','')
df['airline_name'] = df['airline_name'].str.split('|').str[0]

# delete the duplicate flight number in column flight_number behind the strings with a | separator

df['flight_number'] = df['flight_number'].str.split('|').str[0]

df['co2_percentage'] = df['co2_percentage'].str.replace('%','')

# Change object variables to category type

for col in df.columns:
    # Check if the column is of object type
    if df[col].dtype == 'object':
        # If it is, convert it to category type
        df[col] = df[col].astype('category')

        
# Convert text data to numbers

df['from_airport_code'] = df['from_airport_code'].cat.codes
df['from_country'] = df['from_country'].cat.codes
df['dest_airport_code'] = df['dest_airport_code'].cat.codes
df['dest_country'] = df['dest_country'].cat.codes
df['flight_number'] = df['flight_number'].cat.codes

# Define the X columns and Y columns

X = df[['from_airport_code','from_country','dest_airport_code','dest_country','flight_number','duration','stops','co2_emissions','avg_co2_emission_for_this_route','co2_percentage']]
Y = df[['price']]

# Standardization

sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training data and testing data

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# selecting and fitting the model for training
model = RandomForestRegressor()
model.fit(Xtrain, Ytrain)

# saving the trained mode
pickle.dump(model, open('rf_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))
