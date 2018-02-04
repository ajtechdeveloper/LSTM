from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# Frame the data sequence as a supervised learning problem
def timeseries_to_supervised_learning(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Create a differenced series
def difference_series(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

#  Function to Parse date-time values for loading the dataset
def parse_dataset(x):
    return datetime.strptime('201'+x, '%Y-%m')

# Inverse the differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Scale the train and test data to [-1, 1]
def scale_data(train, test):
    # Fit the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # Transform the train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # Transform the test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Invert the scaling for a forecasted value
def invert_scale_value(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# Fit a LSTM network to the training data
def fit_lstm_network(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# Load the dataset
series = read_csv('people_count.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parse_dataset)
# Print the first few rows of the input data
print(series.head())
# Plot the dataset
series.plot()
pyplot.show()
# Transform the data to be stationary
raw_values = series.values
difference_values = difference_series(raw_values, 1)

# Transform the data to a supervised learning problem
supervised_learning = timeseries_to_supervised_learning(difference_values, 1)
supervised_values = supervised_learning.values

# Split the data into train dataset and test dataset
train, test = supervised_values[0:-12], supervised_values[-12:]

# Transform the scale of the data
scaler, train_scaled, test_scaled = scale_data(train, test)

# Fit the LSTM model
lstm_model = fit_lstm_network(train_scaled, 1, 200, 3)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# Rolling Forecast/Walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # Make a one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # Invert scaling
    yhat = invert_scale_value(scaler, X, yhat)
    # Invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # Store the forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month:%d, Expected Value:%f, Predicted Value:%f' % (i+1, expected, yhat))

# Report the performance of the LSTM model using RMSE
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('RMSE: %.3f' % rmse)
# Line plot of observed values vs predicted values
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()