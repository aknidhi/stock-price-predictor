import pandas as pd
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('prices.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

X = data[['Year', 'Month', 'Day']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

last_date = data['Date'].max()
next_date = last_date + pd.DateOffset(days=1)
next_data = pd.DataFrame({'Year': [next_date.year],
                          'Month': [next_date.month],
                          'Day': [next_date.day]})
next_prediction = model.predict(next_data)
print('Predicted closing price for the next day:', next_prediction[0])

trace1 = go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='blue'))
trace2 = go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted', line=dict(color='red'))

data = [trace1, trace2]

layout = dict(title='Actual vs. Predicted Closing Prices',
              xaxis=dict(title='Date'),
              yaxis=dict(title='Closing Price'),
              hovermode='closest')

fig = dict(data=data, layout=layout)
go.Figure(fig).show()
