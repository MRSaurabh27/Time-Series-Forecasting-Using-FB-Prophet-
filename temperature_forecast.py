import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv('MaunaLoaDailyTemps.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[["DATE", "AvgTemp"]]
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# Plot the data
plt.figure(figsize=(18, 6))
plt.plot(df['ds'], df['y'])
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('Mauna Loa Daily Temperatures')
plt.show()

# Split data into train and test sets
train = df.iloc[:len(df) - 365]
test = df.iloc[len(df) - 365:]

# Initialize and fit the model
m = Prophet()
m.fit(train)

# Make future dataframe and predict
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Plot forecast and components
plot_plotly(m, forecast)
plot_components_plotly(m, forecast)

# Calculate RMSE
predictions = forecast.iloc[-365:]['yhat']
print("Root Mean Squared Error between actual and predicted values: ", rmse(predictions, test['y']))
print("Mean Value of Test Dataset:", test['y'].mean())
