# Encode categorical data

from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data):

    # Encode categorical data
    categorical_features = 'Type'

    # Label encoding
    LE = LabelEncoder()
    data[categorical_features] = LE.fit_transform(data[categorical_features])


    # Extract features and target
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']


    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    Numerical_features = ['Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    # Standard Scaler for feature scaling numerical data only
    Scaler = StandardScaler()
    X_train_df[Numerical_features] = Scaler.fit_transform(X_train_df[Numerical_features])
    X_test_df[Numerical_features] = Scaler.transform(X_test_df[Numerical_features])
    
    return X_train_df , X_test_df, y_train, y_test