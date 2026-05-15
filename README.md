# Ex.No: 6               HOLT WINTERS METHOD
### Date: 13/05/2026
### Name : THIRUMALAI K
### REG_NO : 212224240176



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("user_behavior_timeseries.csv")

data.columns = data.columns.str.strip()


data['Date'] = pd.to_datetime(data['Date'])


data.set_index('Date', inplace=True)


daily_data = data.groupby('Date').mean(numeric_only=True)


col = "App Usage Time"


np.random.seed(10)

daily_data[col] = (
    daily_data[col]
    + np.random.normal(0, 2, len(daily_data))
)


scaler = MinMaxScaler()

scaled = scaler.fit_transform(
    daily_data[[col]]
)

scaled_series = pd.Series(
    scaled.flatten(),
    index=daily_data.index
)


scaled_series = scaled_series + 1


plt.figure(figsize=(10,5))

plt.plot(
    scaled_series,
    label='Original Data'
)

plt.title("Original Data")
plt.xlabel("Date")
plt.ylabel("Usage")

plt.legend()
plt.grid(True)

plt.show()


decomposition = seasonal_decompose(
    scaled_series,
    model='additive',
    period=7
)

fig = decomposition.plot()

fig.set_size_inches(10,8)

plt.show()


train_size = int(len(scaled_series) * 0.8)

train_data = scaled_series[:train_size]

test_data = scaled_series[train_size:]


model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()


test_predictions = model.forecast(
    steps=len(test_data)
)

plt.figure(figsize=(10,5))

plt.plot(
    train_data,
    label='train_data'
)

plt.plot(
    test_predictions,
    label='test_predictions_add'
)

plt.plot(
    test_data,
    label='test_data'
)

plt.legend()

plt.title("Visual evaluation")

plt.xlabel("Date")
plt.ylabel("Usage")

plt.grid(True)

plt.show()


rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions
    )
)

print("RMSE =", rmse)


final_model = ExponentialSmoothing(
    scaled_series,
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()


future_predictions = final_model.forecast(15)


plt.figure(figsize=(10,5))

plt.plot(
    scaled_series,
    label='Original Data'
)

plt.plot(
    future_predictions,
    label='Future Forecast'
)

plt.legend()

plt.title("Future Prediction")

plt.xlabel("Date")
plt.ylabel("Usage")

plt.grid(True)

plt.show()
```

### OUTPUT:


TEST_PREDICTION

<img width="875" height="470" alt="image" src="https://github.com/user-attachments/assets/c81e3d5b-cd5f-41af-8a1a-8dc4a0fa9173" />
<img width="900" height="751" alt="image" src="https://github.com/user-attachments/assets/f626fab4-255b-4a40-9401-2c70ec7e5e3e" />


FINAL_PREDICTION
<img width="875" height="470" alt="image" src="https://github.com/user-attachments/assets/58390780-fef0-4e1b-9905-289774c8ea99" />
<img width="851" height="470" alt="image" src="https://github.com/user-attachments/assets/b8ba620f-566b-479d-bf2b-2355c52d839b" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
