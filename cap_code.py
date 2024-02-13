import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Function to process a dataframe
def process_dataframe(df):
    label_encoders = {}
    scalers = {}  # Dictionary to store scaler objects for each numerical column
    numerical_columns = []
    categorical_columns = []
    temporal_columns = []

    for column in df.columns:
        # Check if the column is numerical
        if np.issubdtype(df[column].dtype, np.number):
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler  # Store the scaler object
            numerical_columns.append(column)
        # Check if the column is categorical
        elif df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
            categorical_columns.append(column)
        # Check if the column is datetime
        elif np.issubdtype(df[column].dtype, np.datetime64):
            df[f'{column}_year'] = df[column].dt.year
            df[f'{column}_month'] = df[column].dt.month
            df[f'{column}_day'] = df[column].dt.day
            df.drop(column, axis=1, inplace=True)
            temporal_columns.append(column)
    return df, scalers, numerical_columns, categorical_columns, temporal_columns

# Function to revert scaling
def revert_scaling(df, column, scalers):
    if column in scalers:
        df[f'{column}_original'] = scalers[column].inverse_transform(df[[column]])
    else:
        print(f"Scaler for column {column} not found.")
    return df

processed_df, scalers, numerical_columns, categorical_columns, temporal_columns = process_dataframe(train_set)
pprint(processed_df.columns)