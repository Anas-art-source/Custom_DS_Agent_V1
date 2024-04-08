
import pandas as pd
from xgboost import XGBRegressor
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def xgboost_model(filtered_dataframe, forecast_start_date, forecast_end_date):
    """
    Time series forecasting using XGBoost model.

    Args:
    datapath (str): Path to the CSV file containing the data.
    date_start (str): Start date for prediction in 'YYYY-MM-DD' format.
    date_end (str): End date for prediction in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing the predicted values.
    """

    metrics = {}
    # Read data
    data = filtered_dataframe
    data = data[['Sales', "Date"]]


    # Preprocess data (if required)
    # Example: Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data["Sales"] = data["Sales"].astype(float)

    data.sort_values('Date', inplace=True)
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday

    numerical_features = ['Year', 'Month', 'Day', 'Weekday']
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
            ])
    X = data[numerical_features]

    xgb_regressor = XGBRegressor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_regressor)])

    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics['mse'] = mean_squared_error(y_test,predictions)


    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    date_dataframe = pd.DataFrame({'Date': date_range})
    results = agent_inference(pipeline,date_dataframe)
    metrics['results'] = results
    metrics['average_forcast'] = np.average(results)
    metrics['total_forecast'] = np.sum(results)
    print(metrics)
    return metrics

def agent_inference(pipeline,date_dataframe):

    df = date_dataframe.copy()
    print(df.head())
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    X = df.copy()
    predictions = pipeline.predict(X)
    plt.style.use('dark_background')
    plt.plot(df['Date'], predictions, color='cyan')  # Cyan stands out on a dark background
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.tick_params(colors='white', which='both')  # Change the colors of the tick marks to white
    plt.tight_layout()
    return predictions


