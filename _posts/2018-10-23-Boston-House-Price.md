---
title: Predicting Boston House Pricing
date: 2018-10-23
tags: 
  - Machine Learning
  - Neural Network
  - Regression
header:
  image: "/images/Boston House Price/boston.jpg"
  teaser: "/images/Boston House Price/boston.jpg"
excerpt: "Machine Learning, Neural Network, Regression"
mathjax: "true"
---

Boston house price dataset describes properties of houses in Boston suburbs and is concerned with modeling the price of houses in those suburbs in thousands of dollars. 
<br/>
<br/>
The input variables that describle the properties of a given Boston suburb are:
1. CRIM — per capita crime rate by town.
2. ZN — proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS — proportion of non-retail business acres per town.
4. CHAS — Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5. NOX — nitrogen oxides concentration (parts per 10 million).
6. RM — average number of rooms per dwelling.
7. AGE — proportion of owner-occupied units built prior to 1940.
8. DIS — weighted mean of distances to five Boston employment centers.
9. RAD — index of accessibility to radial highways.
10. TAX — full-value property-tax rate per $10,000.
11. PTRATIO — pupil-teacher ratio by town.
12. B — 1000(Bk — 0.63)² where Bk is the proportion of blacks by town.
13. LSTAT — % lower status of the population.
14. MEDV — median value of owner-occupied homes in $1000s.
<br/>

The aim is to predict median value of a given house in the area, which is the target variable (MEDV).

## 1) Setup
### Load packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load & Viewing the Data
```python
# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, delim_whitespace=True, names=names)
dataset.head(10)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.3e-03</td>
      <td>18.0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.6</td>
      <td>65.2</td>
      <td>4.1</td>
      <td>1</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.9</td>
      <td>5.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.7e-02</td>
      <td>0.0</td>
      <td>7.1</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.4</td>
      <td>78.9</td>
      <td>5.0</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.9</td>
      <td>9.1</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.7e-02</td>
      <td>0.0</td>
      <td>7.1</td>
      <td>0</td>
      <td>0.5</td>
      <td>7.2</td>
      <td>61.1</td>
      <td>5.0</td>
      <td>2</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.8</td>
      <td>4.0</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.2e-02</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>0</td>
      <td>0.5</td>
      <td>7.0</td>
      <td>45.8</td>
      <td>6.1</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.6</td>
      <td>2.9</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.9e-02</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>0</td>
      <td>0.5</td>
      <td>7.1</td>
      <td>54.2</td>
      <td>6.1</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.9</td>
      <td>5.3</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0e-02</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.4</td>
      <td>58.7</td>
      <td>6.1</td>
      <td>3</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.1</td>
      <td>5.2</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.8e-02</td>
      <td>12.5</td>
      <td>7.9</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.0</td>
      <td>66.6</td>
      <td>5.6</td>
      <td>5</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>395.6</td>
      <td>12.4</td>
      <td>22.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.4e-01</td>
      <td>12.5</td>
      <td>7.9</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>96.1</td>
      <td>6.0</td>
      <td>5</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>396.9</td>
      <td>19.1</td>
      <td>27.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.1e-01</td>
      <td>12.5</td>
      <td>7.9</td>
      <td>0</td>
      <td>0.5</td>
      <td>5.6</td>
      <td>100.0</td>
      <td>6.1</td>
      <td>5</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.6</td>
      <td>29.9</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.7e-01</td>
      <td>12.5</td>
      <td>7.9</td>
      <td>0</td>
      <td>0.5</td>
      <td>6.0</td>
      <td>85.9</td>
      <td>6.6</td>
      <td>5</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.7</td>
      <td>17.1</td>
      <td>18.9</td>
    </tr>
  </tbody>
</table>

## 2) Exploratory data analysis
### View the shape of the dataset
```python
dataset.shape
```
{% highlight text %}
(506, 14)
{% endhighlight %} 
This confirms that the data has 14 attributes and 506 instances to work with.

### View the data types of the dataset
```python
dataset.dtypes
```
{% highlight text %}
CRIM       float64
ZN         float64
INDUS      float64
CHAS         int64
NOX        float64
RM         float64
AGE        float64
DIS        float64
RAD          int64
TAX        float64
PTRATIO    float64
B          float64
LSTAT      float64
MEDV       float64
dtype: object
{% endhighlight %} 
Most of the attributes are numeric, mostly real values (float) and some have been interpreted as integers (int).

### View the description of the dataset
```python
pd.set_option('precision', 1)
dataset.describe()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.1e+02</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>5.1e+02</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.6e+00</td>
      <td>11.4</td>
      <td>11.1</td>
      <td>6.9e-02</td>
      <td>0.6</td>
      <td>6.3</td>
      <td>68.6</td>
      <td>3.8</td>
      <td>9.5</td>
      <td>408.2</td>
      <td>18.5</td>
      <td>356.7</td>
      <td>12.7</td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.6e+00</td>
      <td>23.3</td>
      <td>6.9</td>
      <td>2.5e-01</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>28.1</td>
      <td>2.1</td>
      <td>8.7</td>
      <td>168.5</td>
      <td>2.2</td>
      <td>91.3</td>
      <td>7.1</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.3e-03</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0e+00</td>
      <td>0.4</td>
      <td>3.6</td>
      <td>2.9</td>
      <td>1.1</td>
      <td>1.0</td>
      <td>187.0</td>
      <td>12.6</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.2e-02</td>
      <td>0.0</td>
      <td>5.2</td>
      <td>0.0e+00</td>
      <td>0.4</td>
      <td>5.9</td>
      <td>45.0</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>279.0</td>
      <td>17.4</td>
      <td>375.4</td>
      <td>6.9</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.6e-01</td>
      <td>0.0</td>
      <td>9.7</td>
      <td>0.0e+00</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>77.5</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>330.0</td>
      <td>19.1</td>
      <td>391.4</td>
      <td>11.4</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.7e+00</td>
      <td>12.5</td>
      <td>18.1</td>
      <td>0.0e+00</td>
      <td>0.6</td>
      <td>6.6</td>
      <td>94.1</td>
      <td>5.2</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.2</td>
      <td>17.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.9e+01</td>
      <td>100.0</td>
      <td>27.7</td>
      <td>1.0e+00</td>
      <td>0.9</td>
      <td>8.8</td>
      <td>100.0</td>
      <td>12.1</td>
      <td>24.0</td>
      <td>711.0</td>
      <td>22.0</td>
      <td>396.9</td>
      <td>38.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
There seems to be some differing of scales.

### View the correlation between the variables on a heatmap
```python
plt.figure(figsize=(20,10)) 
sns.heatmap(dataset.corr(), annot=True) 
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Boston House Price/heatmap.png" alt="">
Many attributes have a strong correlation:
* INDUS and NOX (0.76)
* INDUS and DIS (-0.71)
* INDUS and TAX (0.72)
* NOX and AGE (0.73)
* NOX and DIS (-0.77)
* LSTAT and target variable MEDV (-0.74)

### View a pairplot on the first 7 features
```python
sns.pairplot(dataset, kind="reg", vars = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE'])
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Boston House Price/pairplot.png" alt="">

## 3) Training
### Prepare dataset for training and testing
Defining the target labels
```python
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
```
Spliting the data into 8:2 for training and testing
```python
from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=5)
```

### Import libraries
```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
```

### Modeling
Selecting a suite of different algorithms capable of working on classification problem.
* **Linear Algorithms:** Logistic Regression(LR), Ridge Regression(Ridge), Lasso Regression(LASSO) and ElasticNet(EN).
* **Nonlinear Algorithms:** Classification and Regression Trees(CART), Support Vector Regression(SVR), and k-Nearest Neighbors(KNN).

```python
models = []
models.append(('LR', LinearRegression()))
models.append(('RIDGE', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
```

Setting a 10-fold cross validation and evaluate using the neg_mean_squared_error scoring metric.
```python
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
```
Displaying the mean and standard deviation of MSE for each algorithm.
```python
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```
{% highlight text %}
LR: -25.223971 (8.769171)
RIDGE: -25.260197 (9.318537)
LASSO: -29.176068 (8.931361)
EN: -28.088055 (8.765610)
KNN: -41.677270 (15.751727)
CART: -22.241501 (11.690218)
SVR: -85.946141 (17.581121)
{% endhighlight %} 

### Viewing the distribution values on a boxplot
```python
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Boston House Price/boxplot.png" alt="">
<br/>
Seems like CART has the lowest MSE, followed by LR. Differing scales of the data might be having an effect on the accuracy of the algorithms such as SVR and KNN.

### Feature scaling
Suspecting that varied distributions of the features may be impacting the skill of some of the algorithms, we perform a data normalization, such that each attribute has a mean value of 0 and a standard deviation of 1.
```python
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', StandardScaler()),('RIDGE', Ridge())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```
{% highlight text %}
ScaledLR: -25.223971 (8.769171)
ScaledLASSO: -30.450409 (10.412011)
ScaledRIDGE: -25.197084 (8.841289)
ScaledEN: -32.387159 (11.796724)
ScaledKNN: -24.495927 (11.884086)
ScaledCART: -25.209998 (20.689530)
ScaledSVR: -32.657487 (15.173412)
{% endhighlight %} 

### Viewing the distribution values on a boxplot
```python
fig = plt.figure()
fig = plt.figure(figsize=(8.5,4))
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Boston House Price/boxplot2.png" alt="">
<br/>
Seems like scaling did have an effect on SVR and KNN. From the boxplot, KNN has a tight distribution of error and also has the lowest score.

## 3b) Ensemble Methods
Looking into ensemble methods could improve the performance of algorithms on this problem. We will evaluate four different ensemble machine learning algorithms.
* **Boosting Methods:** AdaBoost(AB) and Gradient Boosting(GBM).
* **Bagging Methods:** Random Forests(RF) and Extra Trees(ET).
<br/>

Using the same settings, a 10-fold cross validation and pipelines that standardize the training data for each fold.
### Import libraries
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
```
### Modeling
```python
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```
{% highlight text %}
ScaledAB: -13.046332 (4.748033)
ScaledGBM: -9.280277 (3.706627)
ScaledRF: -14.558409 (7.477231)
ScaledET: -11.134913 (4.719559)
{% endhighlight %} 

Viewing the distribution values on a boxplot
```python
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Boston House Price/boxplot3.png" alt="">
<br/>
The results generate better scores than the linear and nonlinear algorithms in previous part. Gradient Boosting has the lowest score and better distribution of error.

## 3c) Neural Network
Looking into neural network to see whether it can improve the performance of algorithms on the regression problem.
### Import libraries
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
```

### Modeling
Create the Keras model and evaluate with scikit-learn by using a wrapper object provided by the Keras library. The Keras wrapper class require a function as an argument and this function is responsible for creating the neural network model to be evaluated. 
<br/>
The model has two hidden layers, each with 32 units(no. of neurons) and 13 input atttributes. Rectifier activation function is used for the hidden layer and the network ends with a single unit with no activation function as this is a regression problem. The model will be compiled with the Adam optimization metric and a mean squared error loss function.
```python
def keras_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```
### Evaluate the model with standardized dataset
Using scikit-learn's Pipeline to perform the standardization and the Keras wrapper object, KerasRegressor as a regression estimator with parameters 100 no. of epochs and batch size of 5.
```python
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=keras_model, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```
{% highlight text %}
Epoch 1/100
455/455 [==============================] - 3s 7ms/step - loss: 569.2438
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 185.6812
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 42.7919
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 29.1629
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 26.0592
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 23.3229
Epoch 7/100
455/455 [==============================] - 1s 2ms/step - loss: 21.9260
Epoch 8/100
455/455 [==============================] - 1s 2ms/step - loss: 20.7636
Epoch 9/100
455/455 [==============================] - 1s 2ms/step - loss: 19.7329
Epoch 10/100
455/455 [==============================] - 1s 2ms/step - loss: 19.2027
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 18.0710
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 17.2367
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 16.4390
Epoch 14/100
455/455 [==============================] - 1s 2ms/step - loss: 15.7287
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 15.0733
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 14.6425
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 13.8937
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 13.6251
Epoch 19/100
455/455 [==============================] - 1s 2ms/step - loss: 13.2056
Epoch 20/100
455/455 [==============================] - 1s 2ms/step - loss: 12.6973
Epoch 21/100
455/455 [==============================] - 1s 2ms/step - loss: 12.6582
Epoch 22/100
455/455 [==============================] - 1s 3ms/step - loss: 12.1058
Epoch 23/100
455/455 [==============================] - 1s 2ms/step - loss: 12.0478
Epoch 24/100
455/455 [==============================] - 1s 2ms/step - loss: 11.7273
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 11.3929
Epoch 26/100
455/455 [==============================] - 1s 1ms/step - loss: 11.0915
Epoch 27/100
455/455 [==============================] - 1s 1ms/step - loss: 10.9539
Epoch 28/100
455/455 [==============================] - 1s 1ms/step - loss: 10.6748
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 10.5449
Epoch 30/100
455/455 [==============================] - 1s 1ms/step - loss: 10.5327
Epoch 31/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3605
Epoch 32/100
455/455 [==============================] - 1s 1ms/step - loss: 10.5112
Epoch 33/100
455/455 [==============================] - 1s 1ms/step - loss: 10.1425
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0132
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0828
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5346
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9640
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6833
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5302
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5269
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4254
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3086
Epoch 43/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2618
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4224
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1985
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1032
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0730
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6949
Epoch 49/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0803
Epoch 50/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9341
Epoch 51/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5511
Epoch 52/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0862
Epoch 53/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8607
Epoch 54/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7151
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 8.4899
Epoch 56/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5086
Epoch 57/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5338
Epoch 58/100
455/455 [==============================] - 1s 1ms/step - loss: 8.4395
Epoch 59/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2676
Epoch 60/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2867
Epoch 61/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2003
Epoch 62/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0303
Epoch 63/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1339
Epoch 64/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3379
Epoch 65/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0155
Epoch 66/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2054
Epoch 67/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8415
Epoch 68/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8787
Epoch 69/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7606
Epoch 70/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8500
Epoch 71/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6308
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5910
Epoch 73/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4585
Epoch 74/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9202
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6600
Epoch 76/100
455/455 [==============================] - 1s 2ms/step - loss: 7.4936
Epoch 77/100
455/455 [==============================] - 1s 2ms/step - loss: 7.4672
Epoch 78/100
455/455 [==============================] - 1s 2ms/step - loss: 7.3695
Epoch 79/100
455/455 [==============================] - 1s 2ms/step - loss: 7.4306
Epoch 80/100
455/455 [==============================] - 1s 2ms/step - loss: 7.4044
Epoch 81/100
455/455 [==============================] - 1s 2ms/step - loss: 7.4252A
Epoch 82/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2535
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3969
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2190
Epoch 85/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2635
Epoch 86/100
455/455 [==============================] - 1s 2ms/step - loss: 7.3862
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1750
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0703
Epoch 89/100
455/455 [==============================] - 0s 1ms/step - loss: 6.8389
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0658
Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0760
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9478
Epoch 93/100
455/455 [==============================] - 1s 2ms/step - loss: 6.9988
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9180
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 6.8428
Epoch 96/100
455/455 [==============================] - 1s 1ms/step - loss: 6.7491
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9112
Epoch 98/100
455/455 [==============================] - 1s 1ms/step - loss: 6.6451
Epoch 99/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9751
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9426
51/51 [==============================] - 1s 16ms/step
Epoch 1/100
455/455 [==============================] - 3s 6ms/step - loss: 547.0717
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 183.2211
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 39.9383
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 28.4535
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 25.7660
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 23.3947
Epoch 7/100
455/455 [==============================] - 1s 1ms/step - loss: 21.9751
Epoch 8/100
455/455 [==============================] - 1s 1ms/step - loss: 20.6482
Epoch 9/100
455/455 [==============================] - 1s 1ms/step - loss: 19.5495
Epoch 10/100
455/455 [==============================] - 1s 1ms/step - loss: 17.9581
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 17.4475
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 16.2292
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 15.0922
Epoch 14/100
455/455 [==============================] - 1s 1ms/step - loss: 14.5497
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 13.9656
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 13.3291
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 12.9627
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 12.4145
Epoch 19/100
455/455 [==============================] - 1s 1ms/step - loss: 12.2620
Epoch 20/100
455/455 [==============================] - 1s 1ms/step - loss: 11.8250
Epoch 21/100
455/455 [==============================] - 1s 1ms/step - loss: 11.9864
Epoch 22/100
455/455 [==============================] - 1s 1ms/step - loss: 11.4355
Epoch 23/100
455/455 [==============================] - 1s 1ms/step - loss: 11.0474
Epoch 24/100
455/455 [==============================] - 1s 1ms/step - loss: 11.5348
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 10.7984
Epoch 26/100
455/455 [==============================] - 1s 1ms/step - loss: 10.9189
Epoch 27/100
455/455 [==============================] - 1s 2ms/step - loss: 10.6951
Epoch 28/100
455/455 [==============================] - 1s 2ms/step - loss: 10.4261
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2561
Epoch 30/100
455/455 [==============================] - 1s 2ms/step - loss: 10.3093
Epoch 31/100
455/455 [==============================] - 1s 2ms/step - loss: 10.1105
Epoch 32/100
455/455 [==============================] - 1s 2ms/step - loss: 10.2740
Epoch 33/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9938
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9522
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 9.7821
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 9.8886
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6948
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6224
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5346
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4538
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5926
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5060
Epoch 43/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6896
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6887
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4851
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4165
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3648
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3511
Epoch 49/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1641
Epoch 50/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2873
Epoch 51/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2287
Epoch 52/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1046
Epoch 53/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1489
Epoch 54/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1390
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9496
Epoch 56/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0155
Epoch 57/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2539
Epoch 58/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8750
Epoch 59/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0055
Epoch 60/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8853
Epoch 61/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7463
Epoch 62/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8184
Epoch 63/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6630
Epoch 64/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6125
Epoch 65/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7189
Epoch 66/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7200
Epoch 67/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6603
Epoch 68/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7517
Epoch 69/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3729
Epoch 70/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3499
Epoch 71/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3124
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5496
Epoch 73/100
455/455 [==============================] - 1s 1ms/step - loss: 8.4513
Epoch 74/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7080
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1344
Epoch 76/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3117
Epoch 77/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3158
Epoch 78/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1006
Epoch 79/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0589
Epoch 80/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2607
Epoch 81/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2218
Epoch 82/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1081
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0734
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0901
Epoch 85/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1353
Epoch 86/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2532
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9221
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8528
Epoch 89/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0301
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8701
Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8080
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9452
Epoch 93/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7692
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7510
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8385
Epoch 96/100

455/455 [==============================] - 1s 1ms/step - loss: 7.5795
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7711
Epoch 98/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5562
Epoch 99/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5824
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6927
51/51 [==============================] - 1s 16ms/step
Epoch 1/100
455/455 [==============================] - 2s 5ms/step - loss: 584.0687
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 212.1279
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 45.0813
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 31.6298
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 28.4323
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 26.3599
Epoch 7/100
455/455 [==============================] - 1s 1ms/step - loss: 24.7778
Epoch 8/100
455/455 [==============================] - 1s 1ms/step - loss: 23.2488
Epoch 9/100
455/455 [==============================] - 1s 1ms/step - loss: 22.2014
Epoch 10/100
455/455 [==============================] - 1s 1ms/step - loss: 20.7752
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 19.5751
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 18.5338
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 17.3301
Epoch 14/100
455/455 [==============================] - 1s 1ms/step - loss: 16.7058
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 15.7088
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 14.7653
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 14.5071
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 13.6810
Epoch 19/100
455/455 [==============================] - 1s 1ms/step - loss: 13.4722
Epoch 20/100
455/455 [==============================] - 1s 1ms/step - loss: 12.9230
Epoch 21/100
455/455 [==============================] - 1s 1ms/step - loss: 12.4497
Epoch 22/100
455/455 [==============================] - 1s 1ms/step - loss: 12.3443
Epoch 23/100
455/455 [==============================] - 1s 1ms/step - loss: 12.0832
Epoch 24/100
455/455 [==============================] - 1s 1ms/step - loss: 11.9453
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 11.7775
Epoch 26/100
455/455 [==============================] - 1s 1ms/step - loss: 11.5981
Epoch 27/100
455/455 [==============================] - 1s 1ms/step - loss: 11.2833
Epoch 28/100
455/455 [==============================] - 1s 1ms/step - loss: 11.1244
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 11.0730
Epoch 30/100
455/455 [==============================] - 1s 1ms/step - loss: 10.8699
Epoch 31/100
455/455 [==============================] - 1s 1ms/step - loss: 10.6133
Epoch 32/100
455/455 [==============================] - 1s 1ms/step - loss: 10.7198
Epoch 33/100
455/455 [==============================] - 1s 1ms/step - loss: 10.5521
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 10.7061
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 10.6011
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 10.4434
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2647
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 10.1563
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 10.1350
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9662
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0352
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9912
Epoch 43/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0457
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 9.8763
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 9.7134
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 9.7039
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9420
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6889
Epoch 49/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3945
Epoch 50/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3054
Epoch 51/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4091
Epoch 52/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3256
Epoch 53/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2066
Epoch 54/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8491
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2222
Epoch 56/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0648
Epoch 57/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0037
Epoch 58/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9507
Epoch 59/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8322
Epoch 60/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9717
Epoch 61/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0431
Epoch 62/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9218
Epoch 63/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6258
Epoch 64/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5227
Epoch 65/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5236
Epoch 66/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6308
Epoch 67/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3663
Epoch 68/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5826
Epoch 69/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3505
Epoch 70/100
455/455 [==============================] - 1s 1ms/step - loss: 8.4499
Epoch 71/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3155
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0739
Epoch 73/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1423
Epoch 74/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2206
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0384
Epoch 76/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0250
Epoch 77/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8279
Epoch 78/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9734
Epoch 79/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0475
Epoch 80/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9546
Epoch 81/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6861
Epoch 82/100
455/455 [==============================] - 1s 2ms/step - loss: 7.8876
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7382
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6437
Epoch 85/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5920
Epoch 86/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7456
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5384
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5705
Epoch 89/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5544
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7235
Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3731
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5408
Epoch 93/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4753
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3734
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3093
Epoch 96/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1485
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2842
Epoch 98/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2822
Epoch 99/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3274
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1524
51/51 [==============================] - 1s 15ms/step
Epoch 1/100
455/455 [==============================] - 3s 6ms/step - loss: 501.4880
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 181.8544
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 41.9020
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 29.5287
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 25.6348
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 22.8425
Epoch 7/100
455/455 [==============================] - 1s 1ms/step - loss: 21.5846
Epoch 8/100
455/455 [==============================] - 1s 1ms/step - loss: 19.9809
Epoch 9/100
455/455 [==============================] - 1s 1ms/step - loss: 18.5429
Epoch 10/100
455/455 [==============================] - 1s 1ms/step - loss: 17.5219
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 16.7191
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 15.8019
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 14.7135
Epoch 14/100
455/455 [==============================] - 1s 1ms/step - loss: 14.2788
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 13.7222
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 12.8810
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 12.7490
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 11.8453
Epoch 19/100
455/455 [==============================] - 1s 1ms/step - loss: 11.4575
Epoch 20/100
455/455 [==============================] - 1s 1ms/step - loss: 11.2901
Epoch 21/100
455/455 [==============================] - 1s 1ms/step - loss: 11.3014
Epoch 22/100
455/455 [==============================] - 1s 1ms/step - loss: 10.6163
Epoch 23/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3057
Epoch 24/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2095
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9298
Epoch 26/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2052
Epoch 27/100
455/455 [==============================] - 1s 1ms/step - loss: 9.8973
Epoch 28/100
455/455 [==============================] - 1s 1ms/step - loss: 9.8593
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6146
Epoch 30/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3754
Epoch 31/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4059
Epoch 32/100
455/455 [==============================] - 1s 1ms/step - loss: 9.3815
Epoch 33/100
455/455 [==============================] - 1s 1ms/step - loss: 9.0996
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9999
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9400
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7392
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8673
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7001
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6537
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6093
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3740
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 8.4647
Epoch 43/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1463
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1756
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3447
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0390
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9861
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0211
Epoch 49/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8325
Epoch 50/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8942
Epoch 51/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7760
Epoch 52/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7428
Epoch 53/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6053
Epoch 54/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6896
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6305
Epoch 56/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5073
Epoch 57/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3990
Epoch 58/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3522
Epoch 59/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2597
Epoch 60/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2019
Epoch 61/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2033
Epoch 62/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2442
Epoch 63/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2280
Epoch 64/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0698
Epoch 65/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0488
Epoch 66/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9734
Epoch 67/100
455/455 [==============================] - 1s 1ms/step - loss: 6.9194
Epoch 68/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0952
Epoch 69/100
455/455 [==============================] - 1s 1ms/step - loss: 6.8755
Epoch 70/100
455/455 [==============================] - 1s 1ms/step - loss: 6.8781
Epoch 71/100
455/455 [==============================] - 1s 1ms/step - loss: 6.7803
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 6.8737
Epoch 73/100
455/455 [==============================] - 1s 1ms/step - loss: 6.6815
Epoch 74/100
455/455 [==============================] - 1s 1ms/step - loss: 6.5111
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 6.6497
Epoch 76/100
455/455 [==============================] - 1s 1ms/step - loss: 6.5725
Epoch 77/100
455/455 [==============================] - 1s 1ms/step - loss: 6.5341
Epoch 78/100
455/455 [==============================] - 1s 1ms/step - loss: 6.4966
Epoch 79/100
455/455 [==============================] - 1s 1ms/step - loss: 6.4146
Epoch 80/100
455/455 [==============================] - 1s 1ms/step - loss: 6.4422
Epoch 81/100
455/455 [==============================] - 1s 1ms/step - loss: 6.3594
Epoch 82/100
455/455 [==============================] - 1s 1ms/step - loss: 6.3505
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 6.4368
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 6.2123
Epoch 85/100
455/455 [==============================] - 1s 1ms/step - loss: 6.1726
Epoch 86/100
455/455 [==============================] - 1s 1ms/step - loss: 6.2601
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 6.1749
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 6.1392
Epoch 89/100
455/455 [==============================] - 1s 1ms/step - loss: 6.1381
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 6.2206

Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 5.9447
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 6.0708
Epoch 93/100
455/455 [==============================] - 1s 1ms/step - loss: 5.9851
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 5.9016
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 5.8593
Epoch 96/100
455/455 [==============================] - 1s 1ms/step - loss: 5.9920
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 5.8447
Epoch 98/100
455/455 [==============================] - 1s 1ms/step - loss: 6.0694
Epoch 99/100
455/455 [==============================] - 1s 1ms/step - loss: 6.0144
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 5.7131
51/51 [==============================] - 1s 17ms/step
Epoch 1/100
455/455 [==============================] - 3s 6ms/step - loss: 525.2103
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 176.8991
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 41.4748
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 29.1822
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 25.2984
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 23.2260
Epoch 7/100
455/455 [==============================] - 1s 1ms/step - loss: 22.0092
Epoch 8/100
455/455 [==============================] - 1s 1ms/step - loss: 20.8753
Epoch 9/100
455/455 [==============================] - 1s 1ms/step - loss: 19.7342
Epoch 10/100
455/455 [==============================] - 1s 1ms/step - loss: 18.6183
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 17.4932
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 16.6544
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 15.8786
Epoch 14/100
455/455 [==============================] - 1s 1ms/step - loss: 15.2774
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 14.5689
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 14.0586
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 13.4937
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 13.3007
Epoch 19/100
455/455 [==============================] - 1s 1ms/step - loss: 12.9998
Epoch 20/100
455/455 [==============================] - 1s 1ms/step - loss: 12.6570
Epoch 21/100
455/455 [==============================] - 1s 1ms/step - loss: 12.5332
Epoch 22/100
455/455 [==============================] - 1s 1ms/step - loss: 12.4234
Epoch 23/100
455/455 [==============================] - 1s 2ms/step - loss: 11.9764
Epoch 24/100
455/455 [==============================] - 1s 2ms/step - loss: 11.6855
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 11.4070
Epoch 26/100
455/455 [==============================] - 1s 2ms/step - loss: 11.2984
Epoch 27/100
455/455 [==============================] - 1s 2ms/step - loss: 11.1186
Epoch 28/100
455/455 [==============================] - 1s 1ms/step - loss: 11.1543
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 10.9087
Epoch 30/100
455/455 [==============================] - 1s 2ms/step - loss: 10.8993
Epoch 31/100
455/455 [==============================] - 1s 1ms/step - loss: 10.9120
Epoch 32/100
455/455 [==============================] - 1s 1ms/step - loss: 10.5037
Epoch 33/100
455/455 [==============================] - 1s 2ms/step - loss: 10.6360
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3868
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3259
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2899
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2729
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3440
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0836
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9001
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3109
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9380
Epoch 43/100
455/455 [==============================] - 1s 1ms/step - loss: 9.7559
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 9.7869
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6164
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6263
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6134
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5583
Epoch 49/100
455/455 [==============================] - 1s 2ms/step - loss: 9.2961
Epoch 50/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2446
Epoch 51/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6739
Epoch 52/100
455/455 [==============================] - 1s 2ms/step - loss: 9.1800
Epoch 53/100
455/455 [==============================] - 1s 2ms/step - loss: 9.1844
Epoch 54/100
455/455 [==============================] - 1s 2ms/step - loss: 9.0930
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9199
Epoch 56/100
455/455 [==============================] - 1s 2ms/step - loss: 9.0329
Epoch 57/100
455/455 [==============================] - 1s 3ms/step - loss: 9.0396
Epoch 58/100
455/455 [==============================] - 2s 3ms/step - loss: 8.8739
Epoch 59/100
455/455 [==============================] - 1s 3ms/step - loss: 9.0284
Epoch 60/100
455/455 [==============================] - 1s 2ms/step - loss: 8.8610
Epoch 61/100
455/455 [==============================] - 1s 2ms/step - loss: 8.8626
Epoch 62/100
455/455 [==============================] - 1s 2ms/step - loss: 8.4299
Epoch 63/100
455/455 [==============================] - 1s 2ms/step - loss: 8.4408
Epoch 64/100
455/455 [==============================] - 1s 2ms/step - loss: 8.3876
Epoch 65/100
455/455 [==============================] - 1s 2ms/step - loss: 8.5280
Epoch 66/100
455/455 [==============================] - 1s 2ms/step - loss: 8.4889
Epoch 67/100
455/455 [==============================] - 1s 2ms/step - loss: 8.3920
Epoch 68/100
455/455 [==============================] - 1s 2ms/step - loss: 8.2836A: 0s - loss: 
Epoch 69/100
455/455 [==============================] - 1s 2ms/step - loss: 8.2286
Epoch 70/100
455/455 [==============================] - 1s 2ms/step - loss: 8.2066
Epoch 71/100
455/455 [==============================] - 1s 2ms/step - loss: 8.1667
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2419
Epoch 73/100
455/455 [==============================] - 1s 2ms/step - loss: 8.4218
Epoch 74/100
455/455 [==============================] - 1s 2ms/step - loss: 8.0784
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9664
Epoch 76/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8771
Epoch 77/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9598
Epoch 78/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8268
Epoch 79/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7552
Epoch 80/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6669
Epoch 81/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6088
Epoch 82/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8224
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8176
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6225
Epoch 85/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5270
Epoch 86/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4976
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4797
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4924
Epoch 89/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4417
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4501
Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2803
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3290
Epoch 93/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3841
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2113
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3421
Epoch 96/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1815
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1484
Epoch 98/100
455/455 [==============================] - 1s 2ms/step - loss: 7.0794
Epoch 99/100
455/455 [==============================] - 1s 2ms/step - loss: 7.1369
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0581
51/51 [==============================] - 1s 20ms/step
Epoch 1/100
455/455 [==============================] - 3s 7ms/step - loss: 498.7908
Epoch 2/100
455/455 [==============================] - 1s 1ms/step - loss: 157.4314
Epoch 3/100
455/455 [==============================] - 1s 1ms/step - loss: 39.1466
Epoch 4/100
455/455 [==============================] - 1s 1ms/step - loss: 30.1004
Epoch 5/100
455/455 [==============================] - 1s 1ms/step - loss: 26.4137
Epoch 6/100
455/455 [==============================] - 1s 1ms/step - loss: 24.4521
Epoch 7/100
455/455 [==============================] - 1s 1ms/step - loss: 22.7989
Epoch 8/100
455/455 [==============================] - 1s 1ms/step - loss: 21.4718
Epoch 9/100
455/455 [==============================] - 1s 1ms/step - loss: 20.3969
Epoch 10/100
455/455 [==============================] - 1s 1ms/step - loss: 19.4870
Epoch 11/100
455/455 [==============================] - 1s 1ms/step - loss: 18.1555
Epoch 12/100
455/455 [==============================] - 1s 1ms/step - loss: 17.3944
Epoch 13/100
455/455 [==============================] - 1s 1ms/step - loss: 16.4192
Epoch 14/100
455/455 [==============================] - 1s 1ms/step - loss: 15.6485
Epoch 15/100
455/455 [==============================] - 1s 1ms/step - loss: 14.8136
Epoch 16/100
455/455 [==============================] - 1s 1ms/step - loss: 14.2511
Epoch 17/100
455/455 [==============================] - 1s 1ms/step - loss: 13.9290
Epoch 18/100
455/455 [==============================] - 1s 1ms/step - loss: 13.4047
Epoch 19/100
455/455 [==============================] - 1s 2ms/step - loss: 12.8580
Epoch 20/100
455/455 [==============================] - 1s 1ms/step - loss: 12.6036
Epoch 21/100
455/455 [==============================] - 1s 1ms/step - loss: 12.2625
Epoch 22/100
455/455 [==============================] - 1s 1ms/step - loss: 12.0892
Epoch 23/100
455/455 [==============================] - 1s 1ms/step - loss: 11.5427
Epoch 24/100
455/455 [==============================] - 1s 1ms/step - loss: 11.5135
Epoch 25/100
455/455 [==============================] - 1s 1ms/step - loss: 11.1676
Epoch 26/100
455/455 [==============================] - 1s 1ms/step - loss: 11.1713
Epoch 27/100
455/455 [==============================] - 1s 1ms/step - loss: 10.9321
Epoch 28/100
455/455 [==============================] - 1s 1ms/step - loss: 11.0647
Epoch 29/100
455/455 [==============================] - 1s 1ms/step - loss: 10.6973
Epoch 30/100
455/455 [==============================] - 1s 1ms/step - loss: 10.2981
Epoch 31/100
455/455 [==============================] - 1s 1ms/step - loss: 10.3347
Epoch 32/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0727: 0s
Epoch 33/100
455/455 [==============================] - 1s 1ms/step - loss: 10.1455
Epoch 34/100
455/455 [==============================] - 1s 1ms/step - loss: 10.0696
Epoch 35/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9519
Epoch 36/100
455/455 [==============================] - 1s 1ms/step - loss: 9.8143
Epoch 37/100
455/455 [==============================] - 1s 1ms/step - loss: 9.9849
Epoch 38/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5491
Epoch 39/100
455/455 [==============================] - 1s 1ms/step - loss: 9.5659
Epoch 40/100
455/455 [==============================] - 1s 1ms/step - loss: 9.6064
Epoch 41/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2990
Epoch 42/100
455/455 [==============================] - 1s 1ms/step - loss: 9.4688
Epoch 43/100
455/455 [==============================] - 1s 2ms/step - loss: 9.4583
Epoch 44/100
455/455 [==============================] - 1s 1ms/step - loss: 9.2109
Epoch 45/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1234
Epoch 46/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1559
Epoch 47/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9906
Epoch 48/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9552
Epoch 49/100
455/455 [==============================] - 1s 1ms/step - loss: 9.1871
Epoch 50/100
455/455 [==============================] - 1s 2ms/step - loss: 8.8915
Epoch 51/100
455/455 [==============================] - 1s 2ms/step - loss: 8.6443
Epoch 52/100
455/455 [==============================] - 1s 2ms/step - loss: 8.8738
Epoch 53/100
455/455 [==============================] - 1s 1ms/step - loss: 8.9684
Epoch 54/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6659
Epoch 55/100
455/455 [==============================] - 1s 1ms/step - loss: 8.8553
Epoch 56/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5221
Epoch 57/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6155
Epoch 58/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7292
Epoch 59/100
455/455 [==============================] - 1s 1ms/step - loss: 8.7234
Epoch 60/100
455/455 [==============================] - 1s 1ms/step - loss: 8.5300TA: 0s - lo
Epoch 61/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3636
Epoch 62/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2025
Epoch 63/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2246
Epoch 64/100
455/455 [==============================] - 1s 1ms/step - loss: 8.6144
Epoch 65/100
455/455 [==============================] - 1s 1ms/step - loss: 8.2274
Epoch 66/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0218
Epoch 67/100
455/455 [==============================] - 1s 1ms/step - loss: 8.1354
Epoch 68/100
455/455 [==============================] - 1s 1ms/step - loss: 8.3243
Epoch 69/100
455/455 [==============================] - 1s 2ms/step - loss: 8.0542
Epoch 70/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8971
Epoch 71/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9877
Epoch 72/100
455/455 [==============================] - 1s 1ms/step - loss: 7.8968
Epoch 73/100
455/455 [==============================] - 1s 1ms/step - loss: 7.9000
Epoch 74/100
455/455 [==============================] - 1s 2ms/step - loss: 7.6107
Epoch 75/100
455/455 [==============================] - 1s 1ms/step - loss: 8.0199
Epoch 76/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7501
Epoch 77/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4263
Epoch 78/100
455/455 [==============================] - 1s 1ms/step - loss: 7.7997
Epoch 79/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5694
Epoch 80/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6087
Epoch 81/100
455/455 [==============================] - 1s 1ms/step - loss: 7.5475
Epoch 82/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3877
Epoch 83/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4599
Epoch 84/100
455/455 [==============================] - 1s 1ms/step - loss: 7.6048
Epoch 85/100

455/455 [==============================] - 1s 1ms/step - loss: 7.3260
Epoch 86/100
455/455 [==============================] - 1s 1ms/step - loss: 7.4328
Epoch 87/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3051
Epoch 88/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3490
Epoch 89/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3311
Epoch 90/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2327
Epoch 91/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2247
Epoch 92/100
455/455 [==============================] - 1s 1ms/step - loss: 7.3276
Epoch 93/100
455/455 [==============================] - 1s 1ms/step - loss: 7.2287
Epoch 94/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1734
Epoch 95/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1224
Epoch 96/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0517
Epoch 97/100
455/455 [==============================] - 1s 1ms/step - loss: 7.1095
Epoch 98/100
455/455 [==============================] - 1s 1ms/step - loss: 7.0147
Epoch 99/100
455/455 [==============================] - 1s 1ms/step - loss: 6.7449
Epoch 100/100
455/455 [==============================] - 1s 1ms/step - loss: 6.8987
51/51 [==============================] - 1s 15ms/step
Epoch 1/100
456/456 [==============================] - 3s 6ms/step - loss: 568.5256
Epoch 2/100
456/456 [==============================] - 1s 1ms/step - loss: 189.9420
Epoch 3/100
456/456 [==============================] - 1s 1ms/step - loss: 47.7176
Epoch 4/100
456/456 [==============================] - 1s 1ms/step - loss: 31.8803
Epoch 5/100
456/456 [==============================] - 1s 1ms/step - loss: 28.1059
Epoch 6/100
456/456 [==============================] - 1s 1ms/step - loss: 25.9047
Epoch 7/100
456/456 [==============================] - 1s 1ms/step - loss: 23.9099
Epoch 8/100
456/456 [==============================] - 1s 1ms/step - loss: 22.3772
Epoch 9/100
456/456 [==============================] - 1s 1ms/step - loss: 21.1823
Epoch 10/100
456/456 [==============================] - 1s 1ms/step - loss: 19.7274
Epoch 11/100
456/456 [==============================] - 1s 1ms/step - loss: 18.9818
Epoch 12/100
456/456 [==============================] - 1s 1ms/step - loss: 18.1683
Epoch 13/100
456/456 [==============================] - 1s 1ms/step - loss: 17.2063
Epoch 14/100
456/456 [==============================] - 1s 1ms/step - loss: 16.2531
Epoch 15/100
456/456 [==============================] - 1s 1ms/step - loss: 15.5113
Epoch 16/100
456/456 [==============================] - 1s 1ms/step - loss: 14.7202
Epoch 17/100
456/456 [==============================] - 1s 1ms/step - loss: 14.1262
Epoch 18/100
456/456 [==============================] - 1s 1ms/step - loss: 13.5998
Epoch 19/100
456/456 [==============================] - 1s 1ms/step - loss: 13.3785
Epoch 20/100
456/456 [==============================] - 1s 1ms/step - loss: 12.6829
Epoch 21/100
456/456 [==============================] - 1s 1ms/step - loss: 12.4238
Epoch 22/100
456/456 [==============================] - 1s 1ms/step - loss: 12.2234
Epoch 23/100
456/456 [==============================] - 1s 1ms/step - loss: 11.9058
Epoch 24/100
456/456 [==============================] - 1s 1ms/step - loss: 11.5915
Epoch 25/100
456/456 [==============================] - 1s 1ms/step - loss: 11.5100
Epoch 26/100
456/456 [==============================] - 1s 1ms/step - loss: 11.2975
Epoch 27/100
456/456 [==============================] - 1s 1ms/step - loss: 11.2362
Epoch 28/100
456/456 [==============================] - 1s 1ms/step - loss: 10.9714
Epoch 29/100
456/456 [==============================] - 1s 1ms/step - loss: 10.8226
Epoch 30/100
456/456 [==============================] - 1s 1ms/step - loss: 10.7062
Epoch 31/100
456/456 [==============================] - 1s 1ms/step - loss: 10.6906
Epoch 32/100
456/456 [==============================] - 1s 1ms/step - loss: 10.4911
Epoch 33/100
456/456 [==============================] - 1s 1ms/step - loss: 10.8248
Epoch 34/100
456/456 [==============================] - 1s 1ms/step - loss: 10.3746
Epoch 35/100
456/456 [==============================] - 1s 1ms/step - loss: 10.4001
Epoch 36/100
456/456 [==============================] - 1s 1ms/step - loss: 10.1546
Epoch 37/100
456/456 [==============================] - 1s 1ms/step - loss: 10.3199
Epoch 38/100
456/456 [==============================] - 1s 1ms/step - loss: 10.1109
Epoch 39/100
456/456 [==============================] - 1s 1ms/step - loss: 10.1187
Epoch 40/100
456/456 [==============================] - 1s 1ms/step - loss: 10.0561
Epoch 41/100
456/456 [==============================] - 1s 1ms/step - loss: 9.7428
Epoch 42/100
456/456 [==============================] - 1s 2ms/step - loss: 9.8864
Epoch 43/100
456/456 [==============================] - 1s 2ms/step - loss: 9.8083
Epoch 44/100
456/456 [==============================] - 1s 2ms/step - loss: 9.7774
Epoch 45/100
456/456 [==============================] - 1s 2ms/step - loss: 9.5676
Epoch 46/100
456/456 [==============================] - 1s 2ms/step - loss: 9.6378
Epoch 47/100
456/456 [==============================] - 1s 2ms/step - loss: 9.6907
Epoch 48/100
456/456 [==============================] - 1s 2ms/step - loss: 9.3910
Epoch 49/100
456/456 [==============================] - 1s 2ms/step - loss: 9.4718
Epoch 50/100
456/456 [==============================] - 1s 2ms/step - loss: 9.4423
Epoch 51/100
456/456 [==============================] - 1s 1ms/step - loss: 9.3935
Epoch 52/100
456/456 [==============================] - 1s 1ms/step - loss: 9.5722
Epoch 53/100
456/456 [==============================] - 1s 1ms/step - loss: 9.5154
Epoch 54/100
456/456 [==============================] - 1s 1ms/step - loss: 9.4235
Epoch 55/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9376
Epoch 56/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1524
Epoch 57/100
456/456 [==============================] - 1s 1ms/step - loss: 9.2151
Epoch 58/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9611
Epoch 59/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9262
Epoch 60/100
456/456 [==============================] - 1s 1ms/step - loss: 9.0597
Epoch 61/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9572
Epoch 62/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9370
Epoch 63/100
456/456 [==============================] - 1s 1ms/step - loss: 8.8026
Epoch 64/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9678
Epoch 65/100
456/456 [==============================] - 1s 1ms/step - loss: 8.9019
Epoch 66/100
456/456 [==============================] - 1s 1ms/step - loss: 8.8226
Epoch 67/100
456/456 [==============================] - 1s 1ms/step - loss: 9.0837
Epoch 68/100
456/456 [==============================] - 1s 1ms/step - loss: 8.7413
Epoch 69/100
456/456 [==============================] - 1s 1ms/step - loss: 8.7053
Epoch 70/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4805
Epoch 71/100
456/456 [==============================] - 1s 1ms/step - loss: 8.6198
Epoch 72/100
456/456 [==============================] - 1s 1ms/step - loss: 8.5685
Epoch 73/100
456/456 [==============================] - 1s 1ms/step - loss: 8.3823
Epoch 74/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2783
Epoch 75/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4004
Epoch 76/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4654
Epoch 77/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2418
Epoch 78/100
456/456 [==============================] - 1s 1ms/step - loss: 8.3030
Epoch 79/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2811
Epoch 80/100
456/456 [==============================] - 1s 1ms/step - loss: 8.3520
Epoch 81/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2115
Epoch 82/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2857
Epoch 83/100
456/456 [==============================] - 1s 1ms/step - loss: 8.1398
Epoch 84/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0780
Epoch 85/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0028
Epoch 86/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0062
Epoch 87/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9215
Epoch 88/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9831
Epoch 89/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0387
Epoch 90/100
456/456 [==============================] - 1s 1ms/step - loss: 8.5132
Epoch 91/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8898
Epoch 92/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8729
Epoch 93/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8290
Epoch 94/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9180
Epoch 95/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6117
Epoch 96/100
456/456 [==============================] - 1s 1ms/step - loss: 7.7217
Epoch 97/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3376
Epoch 98/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8329
Epoch 99/100
456/456 [==============================] - 1s 1ms/step - loss: 7.5098
Epoch 100/100
456/456 [==============================] - 1s 2ms/step - loss: 7.8520
50/50 [==============================] - 1s 18ms/step
Epoch 1/100
456/456 [==============================] - 3s 6ms/step - loss: 565.8993
Epoch 2/100
456/456 [==============================] - 1s 1ms/step - loss: 182.3801
Epoch 3/100
456/456 [==============================] - 1s 1ms/step - loss: 32.8106
Epoch 4/100
456/456 [==============================] - 1s 1ms/step - loss: 21.3299
Epoch 5/100
456/456 [==============================] - 1s 1ms/step - loss: 19.0379
Epoch 6/100
456/456 [==============================] - 1s 1ms/step - loss: 16.8777
Epoch 7/100
456/456 [==============================] - 1s 1ms/step - loss: 15.4311
Epoch 8/100
456/456 [==============================] - 1s 2ms/step - loss: 13.8725
Epoch 9/100
456/456 [==============================] - 1s 2ms/step - loss: 12.7260
Epoch 10/100
456/456 [==============================] - 1s 3ms/step - loss: 11.9084
Epoch 11/100
456/456 [==============================] - 1s 3ms/step - loss: 10.8249
Epoch 12/100
456/456 [==============================] - 1s 2ms/step - loss: 10.2191
Epoch 13/100
456/456 [==============================] - 1s 2ms/step - loss: 9.1849
Epoch 14/100
456/456 [==============================] - 1s 1ms/step - loss: 8.8812
Epoch 15/100
456/456 [==============================] - 1s 2ms/step - loss: 8.2051
Epoch 16/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9252
Epoch 17/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6193
Epoch 18/100
456/456 [==============================] - 1s 1ms/step - loss: 7.1599
Epoch 19/100
456/456 [==============================] - 1s 1ms/step - loss: 7.4032
Epoch 20/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9414
Epoch 21/100
456/456 [==============================] - 1s 1ms/step - loss: 6.7321
Epoch 22/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6634
Epoch 23/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6970
Epoch 24/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4322
Epoch 25/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3358
Epoch 26/100
456/456 [==============================] - 1s 1ms/step - loss: 6.2400
Epoch 27/100
456/456 [==============================] - 1s 1ms/step - loss: 6.2870
Epoch 28/100
456/456 [==============================] - 1s 1ms/step - loss: 6.0669
Epoch 29/100
456/456 [==============================] - 1s 1ms/step - loss: 6.0631
Epoch 30/100
456/456 [==============================] - 1s 1ms/step - loss: 5.8453
Epoch 31/100
456/456 [==============================] - 1s 1ms/step - loss: 6.0661
Epoch 32/100
456/456 [==============================] - 1s 1ms/step - loss: 5.9730
Epoch 33/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3140
Epoch 34/100
456/456 [==============================] - 1s 1ms/step - loss: 6.0345
Epoch 35/100
456/456 [==============================] - 1s 1ms/step - loss: 5.6640
Epoch 36/100
456/456 [==============================] - 1s 1ms/step - loss: 5.7423
Epoch 37/100
456/456 [==============================] - 1s 1ms/step - loss: 5.5302
Epoch 38/100
456/456 [==============================] - 1s 1ms/step - loss: 5.6591
Epoch 39/100
456/456 [==============================] - 1s 1ms/step - loss: 5.6605
Epoch 40/100
456/456 [==============================] - 1s 2ms/step - loss: 5.5401
Epoch 41/100
456/456 [==============================] - 1s 2ms/step - loss: 5.4443
Epoch 42/100
456/456 [==============================] - 1s 1ms/step - loss: 5.4732
Epoch 43/100
456/456 [==============================] - 1s 1ms/step - loss: 5.5074
Epoch 44/100
456/456 [==============================] - 1s 1ms/step - loss: 5.4520
Epoch 45/100
456/456 [==============================] - 1s 1ms/step - loss: 5.5097
Epoch 46/100
456/456 [==============================] - 1s 1ms/step - loss: 5.3278
Epoch 47/100
456/456 [==============================] - 1s 1ms/step - loss: 5.3612
Epoch 48/100
456/456 [==============================] - 1s 1ms/step - loss: 5.2784
Epoch 49/100
456/456 [==============================] - 1s 1ms/step - loss: 5.3591
Epoch 50/100
456/456 [==============================] - 1s 1ms/step - loss: 5.2324
Epoch 51/100
456/456 [==============================] - 1s 2ms/step - loss: 5.3574
Epoch 52/100
456/456 [==============================] - 1s 2ms/step - loss: 5.2286
Epoch 53/100
456/456 [==============================] - 1s 2ms/step - loss: 5.1775
Epoch 54/100
456/456 [==============================] - 1s 2ms/step - loss: 5.2286
Epoch 55/100
456/456 [==============================] - 1s 2ms/step - loss: 5.2144
Epoch 56/100
456/456 [==============================] - 1s 1ms/step - loss: 5.1950
Epoch 57/100
456/456 [==============================] - 1s 2ms/step - loss: 5.2238
Epoch 58/100
456/456 [==============================] - 1s 1ms/step - loss: 5.0257
Epoch 59/100
456/456 [==============================] - 1s 2ms/step - loss: 5.0952
Epoch 60/100
456/456 [==============================] - 1s 2ms/step - loss: 5.0511
Epoch 61/100
456/456 [==============================] - 1s 2ms/step - loss: 5.0759
Epoch 62/100
456/456 [==============================] - 1s 2ms/step - loss: 4.9799
Epoch 63/100
456/456 [==============================] - 1s 2ms/step - loss: 5.0639
Epoch 64/100
456/456 [==============================] - 1s 1ms/step - loss: 5.1276
Epoch 65/100
456/456 [==============================] - 1s 1ms/step - loss: 5.0361
Epoch 66/100
456/456 [==============================] - 1s 2ms/step - loss: 5.0497
Epoch 67/100
456/456 [==============================] - 1s 2ms/step - loss: 5.3701
Epoch 68/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9235
Epoch 69/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9189
Epoch 70/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9926
Epoch 71/100
456/456 [==============================] - 1s 1ms/step - loss: 4.8442
Epoch 72/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9051
Epoch 73/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7751
Epoch 74/100
456/456 [==============================] - 1s 1ms/step - loss: 4.8586
Epoch 75/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7657
Epoch 76/100
456/456 [==============================] - 1s 1ms/step - loss: 4.8150
Epoch 77/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9359
Epoch 78/100
456/456 [==============================] - 1s 1ms/step - loss: 4.8152
Epoch 79/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7634
Epoch 80/100

456/456 [==============================] - 1s 1ms/step - loss: 4.7589
Epoch 81/100
456/456 [==============================] - 1s 1ms/step - loss: 4.8693
Epoch 82/100
456/456 [==============================] - 1s 1ms/step - loss: 5.0660
Epoch 83/100
456/456 [==============================] - 1s 1ms/step - loss: 4.6820
Epoch 84/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9919
Epoch 85/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7983
Epoch 86/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7737
Epoch 87/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7990
Epoch 88/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7331
Epoch 89/100
456/456 [==============================] - 1s 1ms/step - loss: 4.6093
Epoch 90/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7876
Epoch 91/100
456/456 [==============================] - 1s 1ms/step - loss: 4.6716
Epoch 92/100
456/456 [==============================] - 1s 1ms/step - loss: 4.6778
Epoch 93/100
456/456 [==============================] - 1s 1ms/step - loss: 4.6413
Epoch 94/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7568
Epoch 95/100
456/456 [==============================] - 1s 2ms/step - loss: 4.7007
Epoch 96/100
456/456 [==============================] - 1s 2ms/step - loss: 4.7162
Epoch 97/100
456/456 [==============================] - 1s 1ms/step - loss: 4.5173
Epoch 98/100
456/456 [==============================] - 1s 1ms/step - loss: 4.9125
Epoch 99/100
456/456 [==============================] - 1s 1ms/step - loss: 4.4887
Epoch 100/100
456/456 [==============================] - 1s 1ms/step - loss: 4.7334
50/50 [==============================] - 1s 22ms/step
Epoch 1/100
456/456 [==============================] - 3s 7ms/step - loss: 584.7414
Epoch 2/100
456/456 [==============================] - 1s 2ms/step - loss: 183.7957
Epoch 3/100
456/456 [==============================] - 1s 2ms/step - loss: 46.0320
Epoch 4/100
456/456 [==============================] - 1s 2ms/step - loss: 32.6720
Epoch 5/100
456/456 [==============================] - 1s 1ms/step - loss: 28.9797
Epoch 6/100
456/456 [==============================] - 1s 2ms/step - loss: 26.3760
Epoch 7/100
456/456 [==============================] - 1s 2ms/step - loss: 24.0212
Epoch 8/100
456/456 [==============================] - 1s 1ms/step - loss: 21.9821
Epoch 9/100
456/456 [==============================] - 1s 1ms/step - loss: 20.3856
Epoch 10/100
456/456 [==============================] - 1s 1ms/step - loss: 18.7177
Epoch 11/100
456/456 [==============================] - 1s 1ms/step - loss: 17.9984
Epoch 12/100
456/456 [==============================] - 1s 2ms/step - loss: 16.6800
Epoch 13/100
456/456 [==============================] - 1s 1ms/step - loss: 15.7636
Epoch 14/100
456/456 [==============================] - 1s 1ms/step - loss: 14.7337
Epoch 15/100
456/456 [==============================] - 1s 2ms/step - loss: 14.1439
Epoch 16/100
456/456 [==============================] - 1s 1ms/step - loss: 13.5356
Epoch 17/100
456/456 [==============================] - 1s 2ms/step - loss: 12.9756
Epoch 18/100
456/456 [==============================] - 1s 1ms/step - loss: 12.3280
Epoch 19/100
456/456 [==============================] - 1s 1ms/step - loss: 12.1712
Epoch 20/100
456/456 [==============================] - 1s 1ms/step - loss: 11.6905
Epoch 21/100
456/456 [==============================] - 1s 1ms/step - loss: 11.3712
Epoch 22/100
456/456 [==============================] - 1s 1ms/step - loss: 11.2400
Epoch 23/100
456/456 [==============================] - 1s 1ms/step - loss: 10.9726
Epoch 24/100
456/456 [==============================] - 1s 1ms/step - loss: 10.3875
Epoch 25/100
456/456 [==============================] - 1s 2ms/step - loss: 10.7288
Epoch 26/100
456/456 [==============================] - 1s 2ms/step - loss: 10.5374
Epoch 27/100
456/456 [==============================] - 1s 2ms/step - loss: 10.3076
Epoch 28/100
456/456 [==============================] - 1s 2ms/step - loss: 10.0210
Epoch 29/100
456/456 [==============================] - 1s 2ms/step - loss: 10.3644
Epoch 30/100
456/456 [==============================] - 1s 1ms/step - loss: 10.0504
Epoch 31/100
456/456 [==============================] - 1s 1ms/step - loss: 9.6306
Epoch 32/100
456/456 [==============================] - 1s 2ms/step - loss: 9.7019
Epoch 33/100
456/456 [==============================] - 1s 2ms/step - loss: 9.5206
Epoch 34/100
456/456 [==============================] - 1s 1ms/step - loss: 9.5588
Epoch 35/100
456/456 [==============================] - ETA: 0s - loss: 9.542 - 1s 2ms/step - loss: 9.4533
Epoch 36/100
456/456 [==============================] - 1s 2ms/step - loss: 9.2518
Epoch 37/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1843
Epoch 38/100
456/456 [==============================] - 1s 2ms/step - loss: 9.0230
Epoch 39/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1733
Epoch 40/100
456/456 [==============================] - 1s 1ms/step - loss: 9.0576
Epoch 41/100
456/456 [==============================] - 1s 2ms/step - loss: 8.9690
Epoch 42/100
456/456 [==============================] - 1s 2ms/step - loss: 8.8180
Epoch 43/100
456/456 [==============================] - 1s 1ms/step - loss: 8.7325
Epoch 44/100
456/456 [==============================] - 1s 1ms/step - loss: 8.7116
Epoch 45/100
456/456 [==============================] - 1s 2ms/step - loss: 8.8159
Epoch 46/100
456/456 [==============================] - 1s 1ms/step - loss: 8.6669
Epoch 47/100
456/456 [==============================] - 1s 1ms/step - loss: 8.6197
Epoch 48/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4713
Epoch 49/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4635
Epoch 50/100
456/456 [==============================] - 1s 1ms/step - loss: 8.7795
Epoch 51/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4146
Epoch 52/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2995
Epoch 53/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2177
Epoch 54/100
456/456 [==============================] - 1s 1ms/step - loss: 8.1281
Epoch 55/100
456/456 [==============================] - 1s 1ms/step - loss: 8.1825
Epoch 56/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2044
Epoch 57/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0207
Epoch 58/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0806
Epoch 59/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9702
Epoch 60/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9131
Epoch 61/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0708
Epoch 62/100
456/456 [==============================] - 1s 1ms/step - loss: 7.7881
Epoch 63/100
456/456 [==============================] - 1s 1ms/step - loss: 8.0119
Epoch 64/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6463
Epoch 65/100
456/456 [==============================] - 1s 1ms/step - loss: 7.7126
Epoch 66/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6131
Epoch 67/100
456/456 [==============================] - 1s 1ms/step - loss: 7.5996
Epoch 68/100
456/456 [==============================] - 1s 1ms/step - loss: 7.4175
Epoch 69/100
456/456 [==============================] - 1s 1ms/step - loss: 7.5459
Epoch 70/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3673
Epoch 71/100
456/456 [==============================] - 1s 1ms/step - loss: 7.4923
Epoch 72/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3524
Epoch 73/100
456/456 [==============================] - 1s 1ms/step - loss: 7.4666
Epoch 74/100
456/456 [==============================] - 1s 1ms/step - loss: 7.2612
Epoch 75/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3393
Epoch 76/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3161
Epoch 77/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0213
Epoch 78/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9321
Epoch 79/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0501
Epoch 80/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9620A: 0s - los
Epoch 81/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0119
Epoch 82/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0691
Epoch 83/100
456/456 [==============================] - 1s 1ms/step - loss: 6.8569
Epoch 84/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0339
Epoch 85/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9281
Epoch 86/100
456/456 [==============================] - 1s 1ms/step - loss: 6.7250
Epoch 87/100
456/456 [==============================] - 1s 1ms/step - loss: 6.8200
Epoch 88/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6200
Epoch 89/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6971
Epoch 90/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6148
Epoch 91/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6599
Epoch 92/100
456/456 [==============================] - 1s 2ms/step - loss: 6.6272
Epoch 93/100
456/456 [==============================] - 1s 2ms/step - loss: 6.4081
Epoch 94/100
456/456 [==============================] - 1s 2ms/step - loss: 6.4819
Epoch 95/100
456/456 [==============================] - 1s 2ms/step - loss: 6.2534
Epoch 96/100
456/456 [==============================] - 1s 2ms/step - loss: 6.4820
Epoch 97/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3295
Epoch 98/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3386
Epoch 99/100
456/456 [==============================] - 1s 1ms/step - loss: 6.5916
Epoch 100/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4190
50/50 [==============================] - 1s 20ms/step
Epoch 1/100
456/456 [==============================] - 3s 7ms/step - loss: 574.6089
Epoch 2/100
456/456 [==============================] - 1s 1ms/step - loss: 207.5905
Epoch 3/100
456/456 [==============================] - 1s 1ms/step - loss: 42.8691
Epoch 4/100
456/456 [==============================] - 1s 1ms/step - loss: 29.3411
Epoch 5/100
456/456 [==============================] - 1s 1ms/step - loss: 26.3398
Epoch 6/100
456/456 [==============================] - 1s 1ms/step - loss: 24.9373
Epoch 7/100
456/456 [==============================] - 1s 1ms/step - loss: 23.5591
Epoch 8/100
456/456 [==============================] - 1s 1ms/step - loss: 22.2885
Epoch 9/100
456/456 [==============================] - 1s 1ms/step - loss: 21.5287
Epoch 10/100
456/456 [==============================] - 1s 1ms/step - loss: 20.3098
Epoch 11/100
456/456 [==============================] - 1s 1ms/step - loss: 19.6533
Epoch 12/100
456/456 [==============================] - 1s 1ms/step - loss: 18.7114
Epoch 13/100
456/456 [==============================] - 1s 1ms/step - loss: 17.6547
Epoch 14/100
456/456 [==============================] - 1s 1ms/step - loss: 16.9057
Epoch 15/100
456/456 [==============================] - 1s 1ms/step - loss: 16.0593
Epoch 16/100
456/456 [==============================] - 1s 1ms/step - loss: 15.3290
Epoch 17/100
456/456 [==============================] - 1s 1ms/step - loss: 14.6529
Epoch 18/100
456/456 [==============================] - 1s 1ms/step - loss: 14.1372
Epoch 19/100
456/456 [==============================] - 1s 1ms/step - loss: 13.7514
Epoch 20/100
456/456 [==============================] - 1s 1ms/step - loss: 12.9885
Epoch 21/100
456/456 [==============================] - 1s 1ms/step - loss: 12.6308
Epoch 22/100
456/456 [==============================] - 1s 1ms/step - loss: 12.2383
Epoch 23/100
456/456 [==============================] - 1s 1ms/step - loss: 12.0592
Epoch 24/100
456/456 [==============================] - 1s 1ms/step - loss: 11.5486
Epoch 25/100
456/456 [==============================] - 1s 1ms/step - loss: 11.3795
Epoch 26/100
456/456 [==============================] - 1s 1ms/step - loss: 11.1025
Epoch 27/100
456/456 [==============================] - 1s 1ms/step - loss: 11.0307
Epoch 28/100
456/456 [==============================] - 1s 1ms/step - loss: 10.5649
Epoch 29/100
456/456 [==============================] - 1s 1ms/step - loss: 10.6584
Epoch 30/100
456/456 [==============================] - 1s 1ms/step - loss: 10.2913
Epoch 31/100
456/456 [==============================] - 1s 1ms/step - loss: 10.1512
Epoch 32/100
456/456 [==============================] - 1s 1ms/step - loss: 10.0578
Epoch 33/100
456/456 [==============================] - 1s 1ms/step - loss: 9.8331
Epoch 34/100
456/456 [==============================] - 1s 1ms/step - loss: 9.6787
Epoch 35/100
456/456 [==============================] - 1s 1ms/step - loss: 9.5508
Epoch 36/100
456/456 [==============================] - 1s 1ms/step - loss: 9.4931
Epoch 37/100
456/456 [==============================] - 1s 1ms/step - loss: 9.2340
Epoch 38/100
456/456 [==============================] - 1s 1ms/step - loss: 9.2713
Epoch 39/100
456/456 [==============================] - 1s 1ms/step - loss: 9.2289
Epoch 40/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1633
Epoch 41/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1312
Epoch 42/100
456/456 [==============================] - 1s 1ms/step - loss: 9.0457
Epoch 43/100
456/456 [==============================] - 1s 1ms/step - loss: 9.1279
Epoch 44/100
456/456 [==============================] - 1s 1ms/step - loss: 8.8487
Epoch 45/100
456/456 [==============================] - 1s 1ms/step - loss: 8.6156
Epoch 46/100
456/456 [==============================] - 1s 1ms/step - loss: 8.8092
Epoch 47/100
456/456 [==============================] - 1s 1ms/step - loss: 8.3496
Epoch 48/100
456/456 [==============================] - 1s 1ms/step - loss: 8.4611
Epoch 49/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2686
Epoch 50/100
456/456 [==============================] - 1s 1ms/step - loss: 8.2995
Epoch 51/100
456/456 [==============================] - 1s 1ms/step - loss: 8.1474
Epoch 52/100
456/456 [==============================] - 1s 1ms/step - loss: 8.1697
Epoch 53/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8834
Epoch 54/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9429
Epoch 55/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9922
Epoch 56/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9217
Epoch 57/100
456/456 [==============================] - 1s 1ms/step - loss: 7.7758
Epoch 58/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8234
Epoch 59/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8514
Epoch 60/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9765
Epoch 61/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6110
Epoch 62/100
456/456 [==============================] - 1s 1ms/step - loss: 7.6516
Epoch 63/100
456/456 [==============================] - 1s 1ms/step - loss: 7.9543
Epoch 64/100
456/456 [==============================] - 1s 1ms/step - loss: 7.7100
Epoch 65/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3694
Epoch 66/100
456/456 [==============================] - 1s 1ms/step - loss: 7.8683
Epoch 67/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3722
Epoch 68/100
456/456 [==============================] - 1s 1ms/step - loss: 7.2957
Epoch 69/100
456/456 [==============================] - 1s 1ms/step - loss: 7.1712
Epoch 70/100
456/456 [==============================] - 1s 1ms/step - loss: 7.3161
Epoch 71/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0498
Epoch 72/100
456/456 [==============================] - 1s 1ms/step - loss: 7.2716
Epoch 73/100
456/456 [==============================] - 1s 1ms/step - loss: 7.1297
Epoch 74/100

456/456 [==============================] - 1s 1ms/step - loss: 7.3297
Epoch 75/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0511
Epoch 76/100
456/456 [==============================] - 1s 1ms/step - loss: 7.0441
Epoch 77/100
456/456 [==============================] - 1s 1ms/step - loss: 6.8448
Epoch 78/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9298
Epoch 79/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9072
Epoch 80/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9115
Epoch 81/100
456/456 [==============================] - 1s 1ms/step - loss: 6.9031
Epoch 82/100
456/456 [==============================] - 1s 1ms/step - loss: 6.8146
Epoch 83/100
456/456 [==============================] - 1s 1ms/step - loss: 7.2937
Epoch 84/100
456/456 [==============================] - 1s 1ms/step - loss: 6.7212
Epoch 85/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6158
Epoch 86/100
456/456 [==============================] - 1s 1ms/step - loss: 6.7132
Epoch 87/100
456/456 [==============================] - 1s 1ms/step - loss: 6.6483
Epoch 88/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4930
Epoch 89/100
456/456 [==============================] - 1s 1ms/step - loss: 6.5638
Epoch 90/100
456/456 [==============================] - 1s 1ms/step - loss: 6.5567
Epoch 91/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4568
Epoch 92/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4288
Epoch 93/100
456/456 [==============================] - 1s 1ms/step - loss: 6.7731
Epoch 94/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3634
Epoch 95/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4659
Epoch 96/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3137
Epoch 97/100
456/456 [==============================] - 1s 1ms/step - loss: 6.2455
Epoch 98/100
456/456 [==============================] - 1s 1ms/step - loss: 6.3220
Epoch 99/100
456/456 [==============================] - 1s 1ms/step - loss: 6.4257
Epoch 100/100
456/456 [==============================] - 1s 1ms/step - loss: 6.2221
50/50 [==============================] - 1s 21ms/step
Standardized: -24.68 (28.74) MSE
{% endhighlight %} 
The result reports that the mean squared error is 24.68 and the standard deviation is 28.74 from the average across all 10 folds of the cross validation evaluation.

## 4) Tuning
Likely that configuration beyond the default may yield even more accurate models, we investigate tuning the parameters for the two algorithms, KNN and GBM.

### K-Nearest Neighbors
Using a grid search to try a set of different numbers of neighbors and see if it can improve the score. Trials of odd k values from 1 to 21, an arbitary range(Noting that default value of neighbors in KNN is 7). Each k value is then evaluated using 10-fold cross validation on a normalized training dataset.
```python
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```
{% highlight text %}
Best: -21.429139 using {'n_neighbors': 3}
-24.566114 (13.062109) with: {'n_neighbors': 1}
-21.429139 (10.038353) with: {'n_neighbors': 3}
-24.639466 (11.695873) with: {'n_neighbors': 5}
-22.948810 (10.691801) with: {'n_neighbors': 7}
-23.043685 (10.275555) with: {'n_neighbors': 9}
-23.129184 (10.116021) with: {'n_neighbors': 11}
-24.134273 (10.774233) with: {'n_neighbors': 13}
-24.822743 (11.203260) with: {'n_neighbors': 15}
-25.825003 (11.549854) with: {'n_neighbors': 17}
-26.774560 (11.713787) with: {'n_neighbors': 19}
-27.920233 (12.320972) with: {'n_neighbors': 21}
{% endhighlight %} 
The best k neighbor value seems to be 3, with a mean squared error of 21.42.

### Gradient Boosting Machine
Define a parameter grid with range of values from 50 to 400 in increments of 50 (Noting that the default number of boosting stages in GBM is 100). Each boosting stage is then evaluated using 10-fold cross validation on a normalized training dataset. Often, the larger the number of boosting stages, the better the performance but the longer the training time.
```python
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```
{% highlight text %}
Best: -8.861772 using {'n_estimators': 150}
-9.478144 (2.991588) with: {'n_estimators': 50}
-9.016908 (3.372888) with: {'n_estimators': 100}
-8.861772 (3.402912) with: {'n_estimators': 150}
-8.863078 (3.531712) with: {'n_estimators': 200}
-8.895107 (3.701004) with: {'n_estimators': 250}
-8.967938 (3.839772) with: {'n_estimators': 300}
-9.019737 (3.874562) with: {'n_estimators': 350}
-9.051707 (3.886895) with: {'n_estimators': 400}
{% endhighlight %} 
The best configuration was n_estimators of 150, resulting in a mean squared error of 8.86, the best so far.

## 5) Evaluation
Using the gradient boosting model and evaluate it on the hold-out validation dataset. Including normalizing the training dataset before training and also scaling the inputs for the validation dataset and then generate predictions.
```python
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=150)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print("1) MSE: {0:.4f}\n".format(mean_squared_error(Y_validation, predictions)))
print("2) R squared: {0:.4f}\n".format(r2_score(Y_validation, predictions)))
```
{% highlight text %}
1) MSE: 9.2771
2) R squared: 0.8815
{% endhighlight %} 
The results shows that the estimated mean squared error is 9.27, which is quite close to the tuned model. The R squared value indicates a high goodness of fit.


