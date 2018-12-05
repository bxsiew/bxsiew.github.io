---
title: Predicting Google Stock Pricing With Historical Data
date: 2018-11-02
tags: 
  - Neural Network
  - Regression
header:
  image: "/images/Google Stock Price/chart.jpg"
  teaser: "/images/Google Stock Price/chart.jpg"
excerpt: "Neural Network, Regression"
mathjax: "true"
---

Working with historical data to predict the future stock price of Google. This dataset is taken from Kaggle, [AMD and GOOGLE Stock Price](https://www.kaggle.com/gunhee/amdgoogle).

The variables in the dataset are:
* Open and Close, represent the starting and final price at which the stock is traded on a particular day.
* High and Low, represent the maximum and minimum price of the share for the day.
* Adj Close, adjusted closing price is a stock's closing price on any given day of trading that has been amended to include any distributions and corporate actions that occurred at any time before the next day's open.
* Volume, is the number of shares or contracts traded in a security or an entire market during a given period of time.

We will be predicting the upward and downward trends of the open google stock price, that is the stock price at the beginning of the financial day. Noting that the market is closed on weekends and public holidays. This is a Regression problem because we are predicting a continuous outcome (the Google Stock Price).

## 1) Setup
### Load packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Load & Viewing the Data
```python
# Importing the training set
dataset_train = pd.read_csv('Google Stock Price Train.csv')
# Getting the real stock price of aug 2018
dataset_test = pd.read_csv('Google Stock Price Test.csv')

dataset_train.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-05-22</td>
      <td>198.528534</td>
      <td>199.524521</td>
      <td>196.196198</td>
      <td>196.946945</td>
      <td>196.946945</td>
      <td>3433700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-05-26</td>
      <td>196.171173</td>
      <td>202.702698</td>
      <td>195.195190</td>
      <td>202.382385</td>
      <td>202.382385</td>
      <td>6202700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-05-27</td>
      <td>203.023026</td>
      <td>206.136139</td>
      <td>202.607605</td>
      <td>202.982986</td>
      <td>202.982986</td>
      <td>6062500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-05-28</td>
      <td>204.544540</td>
      <td>206.016022</td>
      <td>202.507507</td>
      <td>205.405411</td>
      <td>205.405411</td>
      <td>5332200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-05-29</td>
      <td>206.261261</td>
      <td>208.823822</td>
      <td>205.555557</td>
      <td>208.823822</td>
      <td>208.823822</td>
      <td>5291100</td>
    </tr>
  </tbody>
</table>

## 2) Exploratory data analysis
### View the shape of the dataset
```python
dataset_train.shape
```
{% highlight text %}
(2314, 7)
{% endhighlight %} 
We have 2314 records of financial data.

### View the Google's stock price chart for training
The training data is starts from the 22 May 2009 and ends on 31 July 2018.
```python
#setting index as date
dataset_train['Date'] = pd.to_datetime(dataset_train.Date,format='%Y-%m-%d')
dataset_train.index = dataset_train['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(dataset_train['Open'])
plt.title('Google Stock Price - Yearly', fontsize = 15)
plt.xlabel('Year')
plt.ylabel('Google Stock Price')
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/stock.png" alt="">

### View the Google's stock price chart for testing
The testing data will be the whole financial month of August 2018. This will be the stock price to be predicted.
```python
#setting index as date
dataset_test['Date'] = pd.to_datetime(dataset_test.Date,format='%Y-%m-%d')
dataset_test.index = dataset_test['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(dataset_test['Open'], label='Open Price history')
plt.title('Google Stock Price - Aug 2018', fontsize = 15)
plt.xlabel('Year')
plt.ylabel('Google Stock Price')
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/stock2.png" alt="">


## 3) Modeling
### Prepare dataset for training and testing
Defining the target labels, by extracting out all the records under the Open column.
```python
training_set = dataset_train.iloc[:, 1:2].values
real_stock_price = dataset_test.iloc[:, 1:2].values
```

### Feature Scaling
Normalisation feature scaling is recommended when building RNN. Using the MinMaxScaler function from the sklearn library to perform the normalisation.
<br/>
$${\text Normalization}= \frac{X - X_{\text min}}{X_{\text max} - X_{\text min}}$$
<br/>
Feature range is set to (0,1), because all the new scaled values will be between 0 and 1.
```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

### Creating a data structure with 60 timesteps and 1 output
Training the model to be able to predict the stock price at time t+1, based on the previous 60 stock prices.
```python
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
```

### Reshaping
Adding dimensionality to the data structure. Reshaping into input shape required by the RNN in Keras.
```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

### Importing the Keras libraries and packages
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
```

### Creating the RNN model
Initialising the RNN model with a sequence of LSTM layer of 50 neurons and input shape of X_train created in the reshaping step, followed by a dropout layer with dropout rate of 20% and a output dense layer with a single unit with no activation function.
<br/>

The model will be compiled with the Adam optimization metric and a mean squared error loss function, as this is a regression problem.
For fitting the model, it will be trained for 100 epochs, in batch size of 32 (For every 32 stock prices, it will update the weights by forward propagation and then generating an error that is back propagated into the neural network).

```python
# Initialising the RNN
regressor = Sequential()
regressor.add(LSTM(units = 50, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```
{% highlight text %}
Epoch 1/100
2254/2254 [==============================] - 15s 7ms/step - loss: 0.0125
Epoch 2/100
2254/2254 [==============================] - 17s 7ms/step - loss: 0.0021
Epoch 3/100
2254/2254 [==============================] - 16s 7ms/step - loss: 0.0017
Epoch 4/100
2254/2254 [==============================] - 18s 8ms/step - loss: 0.0017
Epoch 5/100
2254/2254 [==============================] - 18s 8ms/step - loss: 0.0013
Epoch 6/100
2254/2254 [==============================] - 18s 8ms/step - loss: 0.0011
Epoch 7/100
2254/2254 [==============================] - 11s 5ms/step - loss: 0.0012
Epoch 8/100
2254/2254 [==============================] - 14s 6ms/step - loss: 0.0011
Epoch 9/100
2254/2254 [==============================] - 16s 7ms/step - loss: 0.0011
Epoch 10/100
2254/2254 [==============================] - 15s 7ms/step - loss: 0.0011
Epoch 11/100
2254/2254 [==============================] - 14s 6ms/step - loss: 8.7146e-04
Epoch 12/100
2254/2254 [==============================] - 14s 6ms/step - loss: 0.0010
Epoch 13/100
2254/2254 [==============================] - 11s 5ms/step - loss: 9.4041e-04
Epoch 14/100
2254/2254 [==============================] - 12s 5ms/step - loss: 8.8615e-04
Epoch 15/100
2254/2254 [==============================] - 12s 5ms/step - loss: 9.2511e-04
Epoch 16/100
2254/2254 [==============================] - 11s 5ms/step - loss: 9.3190e-04
Epoch 17/100
2254/2254 [==============================] - 16s 7ms/step - loss: 8.3385e-04
Epoch 18/100
2254/2254 [==============================] - 11s 5ms/step - loss: 8.4670e-04
Epoch 19/100
2254/2254 [==============================] - 17s 8ms/step - loss: 9.5342e-04
Epoch 20/100
2254/2254 [==============================] - 12s 6ms/step - loss: 0.0010
Epoch 21/100
2254/2254 [==============================] - 13s 6ms/step - loss: 8.6241e-04
Epoch 22/100
2254/2254 [==============================] - 10s 4ms/step - loss: 7.4177e-04
Epoch 23/100
2254/2254 [==============================] - 16s 7ms/step - loss: 7.8867e-04
Epoch 24/100
2254/2254 [==============================] - 10s 4ms/step - loss: 7.5725e-04
Epoch 25/100
2254/2254 [==============================] - 10s 4ms/step - loss: 7.9248e-04
Epoch 26/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.7279e-04
Epoch 27/100
2254/2254 [==============================] - 17s 7ms/step - loss: 6.4754e-04
Epoch 28/100
2254/2254 [==============================] - 10s 4ms/step - loss: 7.2785e-04
Epoch 29/100
2254/2254 [==============================] - 11s 5ms/step - loss: 7.5460e-04
Epoch 30/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.5743e-04
Epoch 31/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.6111e-04
Epoch 32/100
2254/2254 [==============================] - 10s 4ms/step - loss: 5.9395e-04
Epoch 33/100
2254/2254 [==============================] - 10s 4ms/step - loss: 5.6595e-04
Epoch 34/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.4271e-04
Epoch 35/100
2254/2254 [==============================] - 9s 4ms/step - loss: 7.7035e-04
Epoch 36/100
2254/2254 [==============================] - 9s 4ms/step - loss: 6.4927e-04
Epoch 37/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.8631e-04
Epoch 38/100
2254/2254 [==============================] - 9s 4ms/step - loss: 6.6360e-04
Epoch 39/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.9978e-04
Epoch 40/100
2254/2254 [==============================] - 9s 4ms/step - loss: 6.0567e-04
Epoch 41/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.5976e-04
Epoch 42/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.5310e-04
Epoch 43/100
2254/2254 [==============================] - 10s 4ms/step - loss: 6.0196e-04
Epoch 44/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.1869e-04
Epoch 45/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.7660e-04
Epoch 46/100
2254/2254 [==============================] - 10s 4ms/step - loss: 5.3148e-04
Epoch 47/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.3247e-04
Epoch 48/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.1230e-04
Epoch 49/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.3311e-04
Epoch 50/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.9404e-04
Epoch 51/100
2254/2254 [==============================] - 10s 4ms/step - loss: 4.6761e-04
Epoch 52/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.2368e-04
Epoch 53/100
2254/2254 [==============================] - 10s 4ms/step - loss: 5.2032e-04
Epoch 54/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.0123e-04
Epoch 55/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.8538e-04
Epoch 56/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.0737e-04
Epoch 57/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.0824e-04
Epoch 58/100
2254/2254 [==============================] - 10s 4ms/step - loss: 4.9408e-04
Epoch 59/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7324e-04
Epoch 60/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7012e-04
Epoch 61/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7371e-04
Epoch 62/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4469e-04
Epoch 63/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.0811e-04
Epoch 64/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.2099e-04
Epoch 65/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.0001e-04
Epoch 66/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.6915e-04
Epoch 67/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7570e-04
Epoch 68/100
2254/2254 [==============================] - 10s 4ms/step - loss: 4.3574e-04
Epoch 69/100
2254/2254 [==============================] - 9s 4ms/step - loss: 5.1852e-04
Epoch 70/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.5907e-04
Epoch 71/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4451e-04
Epoch 72/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4113e-04
Epoch 73/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.1862e-04
Epoch 74/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7895e-04
Epoch 75/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.7253e-04
Epoch 76/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.3955e-04
Epoch 77/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4898e-04
Epoch 78/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4652e-04
Epoch 79/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4967e-04
Epoch 80/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4691e-04
Epoch 81/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.5056e-04
Epoch 82/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4035e-04
Epoch 83/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4110e-04
Epoch 84/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.5617e-04
Epoch 85/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.6001e-04
Epoch 86/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.8532e-04
Epoch 87/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.1961e-04
Epoch 88/100
2254/2254 [==============================] - 9s 4ms/step - loss: 4.4113e-04
Epoch 89/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.5073e-04
Epoch 90/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.2665e-04
Epoch 91/100
2254/2254 [==============================] - 11s 5ms/step - loss: 3.8613e-04
Epoch 92/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.2036e-04
Epoch 93/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.7794e-04
Epoch 94/100
2254/2254 [==============================] - 10s 4ms/step - loss: 4.2463e-04
Epoch 95/100
2254/2254 [==============================] - 10s 4ms/step - loss: 4.9482e-04
Epoch 96/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.1721e-04
Epoch 97/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.3711e-04
Epoch 98/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.3975e-04
Epoch 99/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.1762e-04
Epoch 100/100
2254/2254 [==============================] - 11s 5ms/step - loss: 4.5311e-04
{% endhighlight %} 

### Getting the predicted stock price of Aug 2018
```python
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```
### Visualising the results
```python
plt.figure(figsize=(12,8)) 
plt.plot(real_stock_price, label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'orange', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/pred.png" alt="">


## 4) 60 Timestep, 2 layers
Using the same settings, add a second LSTM layer and a second Dropout regularisation (return_sequences set to true as this is a stacked LSTM).
```python
# Initialising the RNN
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```
{% highlight text %}
Epoch 1/100
2254/2254 [==============================] - 23s 10ms/step - loss: 0.0097
Epoch 2/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0021
Epoch 3/100
2254/2254 [==============================] - 19s 8ms/step - loss: 0.0018
Epoch 4/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0021
Epoch 5/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0015
Epoch 6/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0015
Epoch 7/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0015
Epoch 8/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0013
Epoch 9/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0014
Epoch 10/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0014
Epoch 11/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0012
Epoch 12/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0013
Epoch 13/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0011
Epoch 14/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0011
Epoch 15/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0011
Epoch 16/100
2254/2254 [==============================] - 19s 9ms/step - loss: 0.0011
Epoch 17/100
2254/2254 [==============================] - 19s 8ms/step - loss: 0.0010
Epoch 18/100
2254/2254 [==============================] - 19s 8ms/step - loss: 9.7959e-04
Epoch 19/100
2254/2254 [==============================] - 19s 8ms/step - loss: 9.8135e-04
Epoch 20/100
2254/2254 [==============================] - 19s 8ms/step - loss: 0.0010
Epoch 21/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.6143e-04
Epoch 22/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.9943e-04
Epoch 23/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.9403e-04
Epoch 24/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.0058e-04
Epoch 25/100
2254/2254 [==============================] - 19s 8ms/step - loss: 9.4341e-04
Epoch 26/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.7610e-04
Epoch 27/100
2254/2254 [==============================] - 19s 8ms/step - loss: 9.2974e-04
Epoch 28/100
2254/2254 [==============================] - 19s 8ms/step - loss: 9.3606e-04
Epoch 29/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.2092e-04
Epoch 30/100
2254/2254 [==============================] - 19s 8ms/step - loss: 8.0920e-04
Epoch 31/100
2254/2254 [==============================] - 19s 8ms/step - loss: 7.6745e-04
Epoch 32/100
2254/2254 [==============================] - 19s 9ms/step - loss: 8.1150e-04
Epoch 33/100
2254/2254 [==============================] - 19s 8ms/step - loss: 6.7653e-04
Epoch 34/100
2254/2254 [==============================] - 19s 8ms/step - loss: 7.0652e-04
Epoch 35/100
2254/2254 [==============================] - 19s 8ms/step - loss: 7.1441e-04
Epoch 36/100
2254/2254 [==============================] - 19s 8ms/step - loss: 7.0913e-04
Epoch 37/100
2254/2254 [==============================] - 19s 8ms/step - loss: 6.5791e-04
Epoch 38/100
2254/2254 [==============================] - 19s 8ms/step - loss: 7.2947e-04
Epoch 39/100
2254/2254 [==============================] - 21s 9ms/step - loss: 7.1467e-04
Epoch 40/100
2254/2254 [==============================] - 20s 9ms/step - loss: 8.6220e-04
Epoch 41/100
2254/2254 [==============================] - 21s 9ms/step - loss: 6.5179e-04
Epoch 42/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.6944e-04
Epoch 43/100
2254/2254 [==============================] - 22s 10ms/step - loss: 6.4108e-04
Epoch 44/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.2510e-04
Epoch 45/100
2254/2254 [==============================] - 24s 11ms/step - loss: 6.6744e-04
Epoch 46/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.4234e-04
Epoch 47/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.8520e-04
Epoch 48/100
2254/2254 [==============================] - 25s 11ms/step - loss: 6.5982e-04
Epoch 49/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.7573e-04
Epoch 50/100
2254/2254 [==============================] - 23s 10ms/step - loss: 7.8931e-04
Epoch 51/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.7463e-04
Epoch 52/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.0428e-04
Epoch 53/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.5159e-04
Epoch 54/100
2254/2254 [==============================] - 24s 11ms/step - loss: 6.6653e-04
Epoch 55/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.0377e-04
Epoch 56/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.9649e-04
Epoch 57/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.9961e-04
Epoch 58/100
2254/2254 [==============================] - 26s 11ms/step - loss: 5.7356e-04
Epoch 59/100
2254/2254 [==============================] - 28s 12ms/step - loss: 6.1476e-04
Epoch 60/100
2254/2254 [==============================] - 24s 11ms/step - loss: 5.6680e-04
Epoch 61/100
2254/2254 [==============================] - 21s 9ms/step - loss: 6.0328e-04
Epoch 62/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.7130e-04
Epoch 63/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.6986e-04
Epoch 64/100
2254/2254 [==============================] - 25s 11ms/step - loss: 6.2032e-04
Epoch 65/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.8592e-04
Epoch 66/100
2254/2254 [==============================] - 20s 9ms/step - loss: 5.8664e-04
Epoch 67/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.8237e-04
Epoch 68/100
2254/2254 [==============================] - 26s 11ms/step - loss: 5.4182e-04
Epoch 69/100
2254/2254 [==============================] - 24s 11ms/step - loss: 5.1365e-04
Epoch 70/100
2254/2254 [==============================] - 24s 11ms/step - loss: 6.3595e-04
Epoch 71/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.9597e-04
Epoch 72/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.1293e-04
Epoch 73/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.9058e-04
Epoch 74/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.6533e-04
Epoch 75/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.7021e-04
Epoch 76/100
2254/2254 [==============================] - 24s 10ms/step - loss: 5.3937e-04
Epoch 77/100
2254/2254 [==============================] - 26s 12ms/step - loss: 5.2816e-04
Epoch 78/100
2254/2254 [==============================] - 32s 14ms/step - loss: 5.1024e-04
Epoch 79/100
2254/2254 [==============================] - 25s 11ms/step - loss: 5.3611e-04
Epoch 80/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.6560e-04
Epoch 81/100
2254/2254 [==============================] - 23s 10ms/step - loss: 6.1973e-04
Epoch 82/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.6534e-04
Epoch 83/100
2254/2254 [==============================] - 24s 11ms/step - loss: 5.2241e-04
Epoch 84/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.7644e-04
Epoch 85/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.7419e-04
Epoch 86/100
2254/2254 [==============================] - 24s 11ms/step - loss: 6.4461e-04
Epoch 87/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.9470e-04
Epoch 88/100
2254/2254 [==============================] - 24s 11ms/step - loss: 4.6464e-04
Epoch 89/100
2254/2254 [==============================] - 19s 9ms/step - loss: 5.1324e-04
Epoch 90/100
2254/2254 [==============================] - 19s 8ms/step - loss: 5.5506e-04
Epoch 91/100
2254/2254 [==============================] - 22s 10ms/step - loss: 4.8305e-04
Epoch 92/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.4446e-04
Epoch 93/100
2254/2254 [==============================] - 23s 10ms/step - loss: 5.2699e-04
Epoch 94/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.4366e-04
Epoch 95/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.8225e-04
Epoch 96/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.8659e-04
Epoch 97/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.3299e-04
Epoch 98/100
2254/2254 [==============================] - 20s 9ms/step - loss: 4.8070e-04
Epoch 99/100
2254/2254 [==============================] - 21s 9ms/step - loss: 5.3787e-04
Epoch 100/100
2254/2254 [==============================] - 24s 11ms/step - loss: 4.9874e-04
{% endhighlight %} 

### Getting the predicted stock price of Aug 2018
```python
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

### Visualising the results
```python
plt.figure(figsize=(12,8)) 
plt.plot(real_stock_price, label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'orange', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/pred2.png" alt="">


## 5) 60 Timestep, 4 layers
This time add a third and fourth LSTM layer and Dropout regularisation, retaining the previous settings.
```python
# Initialising the RNN
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```
{% highlight text %}
Epoch 1/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0167
Epoch 2/100
2254/2254 [==============================] - 44s 20ms/step - loss: 0.0025
Epoch 3/100
2254/2254 [==============================] - 41s 18ms/step - loss: 0.0023
Epoch 4/100
2254/2254 [==============================] - 42s 19ms/step - loss: 0.0021
Epoch 5/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0020
Epoch 6/100
2254/2254 [==============================] - 41s 18ms/step - loss: 0.0021
Epoch 7/100
2254/2254 [==============================] - 44s 19ms/step - loss: 0.0027
Epoch 8/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0018
Epoch 9/100
2254/2254 [==============================] - 44s 20ms/step - loss: 0.0017
Epoch 10/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0016
Epoch 11/100
2254/2254 [==============================] - 46s 21ms/step - loss: 0.0018
Epoch 12/100
2254/2254 [==============================] - 42s 19ms/step - loss: 0.0018
Epoch 13/100
2254/2254 [==============================] - 48s 21ms/step - loss: 0.0015
Epoch 14/100
2254/2254 [==============================] - 52s 23ms/step - loss: 0.0016
Epoch 15/100
2254/2254 [==============================] - 48s 21ms/step - loss: 0.0013
Epoch 16/100
2254/2254 [==============================] - 46s 20ms/step - loss: 0.0013
Epoch 17/100
2254/2254 [==============================] - 50s 22ms/step - loss: 0.0014
Epoch 18/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0013
Epoch 19/100
2254/2254 [==============================] - 49s 22ms/step - loss: 0.0013
Epoch 20/100
2254/2254 [==============================] - 49s 22ms/step - loss: 0.0014
Epoch 21/100
2254/2254 [==============================] - 49s 22ms/step - loss: 0.0011
Epoch 22/100
2254/2254 [==============================] - 48s 21ms/step - loss: 0.0012
Epoch 23/100
2254/2254 [==============================] - 54s 24ms/step - loss: 0.0011
Epoch 24/100
2254/2254 [==============================] - 46s 20ms/step - loss: 0.0012
Epoch 25/100
2254/2254 [==============================] - 43s 19ms/step - loss: 0.0014
Epoch 26/100
2254/2254 [==============================] - 42s 19ms/step - loss: 0.0014
Epoch 27/100
2254/2254 [==============================] - 50s 22ms/step - loss: 0.0011
Epoch 28/100
2254/2254 [==============================] - 53s 24ms/step - loss: 0.0010
Epoch 29/100
2254/2254 [==============================] - 44s 19ms/step - loss: 0.0011
Epoch 30/100
2254/2254 [==============================] - 43s 19ms/step - loss: 0.0011
Epoch 31/100
2254/2254 [==============================] - 47s 21ms/step - loss: 0.0012
Epoch 32/100
2254/2254 [==============================] - 48s 21ms/step - loss: 0.0010
Epoch 33/100
2254/2254 [==============================] - 50s 22ms/step - loss: 0.0011
Epoch 34/100
2254/2254 [==============================] - 46s 20ms/step - loss: 0.0010
Epoch 35/100
2254/2254 [==============================] - 46s 20ms/step - loss: 0.0010
Epoch 36/100
2254/2254 [==============================] - 45s 20ms/step - loss: 9.5280e-04
Epoch 37/100
2254/2254 [==============================] - 47s 21ms/step - loss: 9.7453e-04
Epoch 38/100
2254/2254 [==============================] - 48s 21ms/step - loss: 8.7729e-04
Epoch 39/100
2254/2254 [==============================] - 43s 19ms/step - loss: 9.9251e-04
Epoch 40/100
2254/2254 [==============================] - 43s 19ms/step - loss: 8.4775e-04
Epoch 41/100
2254/2254 [==============================] - 42s 19ms/step - loss: 9.1941e-04
Epoch 42/100
2254/2254 [==============================] - 47s 21ms/step - loss: 9.4173e-04
Epoch 43/100
2254/2254 [==============================] - 43s 19ms/step - loss: 9.6453e-04
Epoch 44/100
2254/2254 [==============================] - 43s 19ms/step - loss: 9.4788e-04
Epoch 45/100
2254/2254 [==============================] - 44s 20ms/step - loss: 9.2993e-04
Epoch 46/100
2254/2254 [==============================] - 43s 19ms/step - loss: 0.0011
Epoch 47/100
2254/2254 [==============================] - 45s 20ms/step - loss: 0.0011
Epoch 48/100
2254/2254 [==============================] - 43s 19ms/step - loss: 8.3628e-04
Epoch 49/100
2254/2254 [==============================] - 42s 19ms/step - loss: 8.4610e-04
Epoch 50/100
2254/2254 [==============================] - 42s 19ms/step - loss: 8.2313e-04
Epoch 51/100
2254/2254 [==============================] - 49s 22ms/step - loss: 8.1417e-04
Epoch 52/100
2254/2254 [==============================] - 45s 20ms/step - loss: 8.3760e-04
Epoch 53/100
2254/2254 [==============================] - 43s 19ms/step - loss: 8.1666e-04
Epoch 54/100
2254/2254 [==============================] - 43s 19ms/step - loss: 8.7949e-04
Epoch 55/100
2254/2254 [==============================] - 44s 20ms/step - loss: 8.2474e-04
Epoch 56/100
2254/2254 [==============================] - 45s 20ms/step - loss: 7.4370e-04
Epoch 57/100
2254/2254 [==============================] - 46s 20ms/step - loss: 8.6703e-04
Epoch 58/100
2254/2254 [==============================] - 44s 19ms/step - loss: 8.0006e-04
Epoch 59/100
2254/2254 [==============================] - 46s 20ms/step - loss: 9.8098e-04
Epoch 60/100
2254/2254 [==============================] - 44s 20ms/step - loss: 8.9601e-04
Epoch 61/100
2254/2254 [==============================] - 46s 20ms/step - loss: 8.2654e-04
Epoch 62/100
2254/2254 [==============================] - 43s 19ms/step - loss: 7.5304e-04
Epoch 63/100
2254/2254 [==============================] - 42s 19ms/step - loss: 6.9636e-04
Epoch 64/100
2254/2254 [==============================] - 43s 19ms/step - loss: 7.0572e-04
Epoch 65/100
2254/2254 [==============================] - 43s 19ms/step - loss: 9.5200e-04
Epoch 66/100
2254/2254 [==============================] - 42s 19ms/step - loss: 8.1569e-04
Epoch 67/100
2254/2254 [==============================] - 54s 24ms/step - loss: 7.4162e-04
Epoch 68/100
2254/2254 [==============================] - 45s 20ms/step - loss: 8.7131e-04
Epoch 69/100
2254/2254 [==============================] - 43s 19ms/step - loss: 8.1044e-04
Epoch 70/100
2254/2254 [==============================] - 43s 19ms/step - loss: 7.0621e-04
Epoch 71/100
2254/2254 [==============================] - 46s 20ms/step - loss: 7.4045e-04
Epoch 72/100
2254/2254 [==============================] - 43s 19ms/step - loss: 7.4376e-04
Epoch 73/100
2254/2254 [==============================] - 40s 18ms/step - loss: 7.7331e-04
Epoch 74/100
2254/2254 [==============================] - 34s 15ms/step - loss: 7.4336e-04
Epoch 75/100
2254/2254 [==============================] - 33s 15ms/step - loss: 8.0720e-04
Epoch 76/100
2254/2254 [==============================] - 33s 15ms/step - loss: 7.3442e-04
Epoch 77/100
2254/2254 [==============================] - 37s 16ms/step - loss: 7.3516e-04
Epoch 78/100
2254/2254 [==============================] - 42s 19ms/step - loss: 6.9743e-04
Epoch 79/100
2254/2254 [==============================] - 42s 19ms/step - loss: 7.4002e-04
Epoch 80/100
2254/2254 [==============================] - 46s 20ms/step - loss: 7.6860e-04
Epoch 81/100
2254/2254 [==============================] - 45s 20ms/step - loss: 7.6494e-04
Epoch 82/100
2254/2254 [==============================] - 43s 19ms/step - loss: 6.6910e-04
Epoch 83/100
2254/2254 [==============================] - 42s 19ms/step - loss: 7.7146e-04
Epoch 84/100
2254/2254 [==============================] - 45s 20ms/step - loss: 7.8184e-04
Epoch 85/100
2254/2254 [==============================] - 49s 22ms/step - loss: 7.2387e-04
Epoch 86/100
2254/2254 [==============================] - 45s 20ms/step - loss: 6.8862e-04
Epoch 87/100
2254/2254 [==============================] - 43s 19ms/step - loss: 5.6815e-04
Epoch 88/100
2254/2254 [==============================] - 45s 20ms/step - loss: 6.6187e-04
Epoch 89/100
2254/2254 [==============================] - 44s 20ms/step - loss: 6.9727e-04
Epoch 90/100
2254/2254 [==============================] - 44s 19ms/step - loss: 8.4382e-04
Epoch 91/100
2254/2254 [==============================] - 42s 19ms/step - loss: 6.9760e-04
Epoch 92/100
2254/2254 [==============================] - 46s 20ms/step - loss: 7.2048e-04
Epoch 93/100
2254/2254 [==============================] - 46s 20ms/step - loss: 8.3656e-04
Epoch 94/100
2254/2254 [==============================] - 43s 19ms/step - loss: 7.0138e-04
Epoch 95/100
2254/2254 [==============================] - 43s 19ms/step - loss: 6.5958e-04
Epoch 96/100
2254/2254 [==============================] - 43s 19ms/step - loss: 6.8683e-04
Epoch 97/100
2254/2254 [==============================] - 54s 24ms/step - loss: 6.9979e-04
Epoch 98/100
2254/2254 [==============================] - 50s 22ms/step - loss: 7.6207e-04
Epoch 99/100
2254/2254 [==============================] - 53s 23ms/step - loss: 6.6170e-04
Epoch 100/100
2254/2254 [==============================] - 49s 22ms/step - loss: 6.6635e-04
{% endhighlight %} 

### Getting the predicted stock price of Aug 2018
```python
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

### Visualising the results
```python
plt.figure(figsize=(12,8)) 
plt.plot(real_stock_price, label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'orange', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Google Stock Price/pred3.png" alt="">

## 6) Conclusion
LSTM model are performing well and it might even be performing better than the traditional ARIMA model. The model with the timesteps of 60 and 1 layer seems to be having the best predictions against the actual stock prices. In reality, stock prices are affected by many other factors that could even improve our model.
<br/>

One way of improving the model is to add some other indicators(having the financial instinct) that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.

