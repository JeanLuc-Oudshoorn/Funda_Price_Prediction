import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Define the list of CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

# Initialize an empty list to store the dataframes
dataframes = []

# Loop through the CSV files and load them into dataframes
for file in csv_files:
    data = pd.read_csv(file)
    dataframes.append(data)

# Concatenate the dataframes vertically
df = pd.concat(dataframes, ignore_index=True)

# Define the outcome variable and the categorical variables
outcome_variable = 'price'
numeric_variables = ['house_age', 'living_area', 'bedroom', 'bathroom']
categorical_variables = ['house_type', 'building_type', 'energy_label', 'zip']

# Preprocess the data
for var in numeric_variables:
    df[var] = df[var].astype('float64')

for var in categorical_variables:
    le = LabelEncoder()
    df[var] = le.fit_transform(df[var])

# Subset X for only the variables in numeric_variables and categorical_variables
X = df[[var for var in df.columns if var in numeric_variables + categorical_variables]]
y = df['price']

# Train the model on all data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Define the custom row
custom_row = pd.DataFrame([{'house_type': 0, 'building_type': 0, 'bedroom': 2, 'bathroom': 1, 'living_area': 89,
                            'energy_label': 0, 'zip': 0, 'house_age': 21}])

# Make a prediction on the custom row
prediction = model.predict(custom_row)

# Print the prediction
print("Prediction for the custom row:", prediction)
