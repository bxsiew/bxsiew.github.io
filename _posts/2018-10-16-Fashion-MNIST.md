---
title: Fashion MNIST
date: 2018-10-16
tags: 
  - machine learning
  - data science
header:
  image: ""
excerpt: "Machine Learning, Data Science"
mathjax: "true"
---
Fashion-MNIST is a dataset of Zalando's article images consisting of 70,000 images divided into 60,000 training and 10,000 testing samples.
Each dataset sample consists of a 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:
0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. 
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/fashionmnist.jpg" alt="">

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
# dataframes creation for both training and testing datasets 
fashion_train_df = pd.read_csv('fashion-mnist_train.csv',sep=',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')
fashion_train_df.head()
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>
The dataset have 785 columns. The first column consists of the class labels(0-9), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

## 2) Exploratory data analysis
View the shape of the dataframe
```python
fashion_train_df.shape
```
{% highlight text %}
(60000, 785)
{% endhighlight %}
```python
fashion_test_df.shape
```
{% highlight text %}
(10000, 785)
{% endhighlight %}

Create training and testing arrays so that we can visualize the data
```python
training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype='float32')
```
Viewing the arrays
```python
training
```
{% highlight text %}
array([[2., 0., 0., ..., 0., 0., 0.],
       [9., 0., 0., ..., 0., 0., 0.],
       [6., 0., 0., ..., 0., 0., 0.],
       ...,
       [8., 0., 0., ..., 0., 0., 0.],
       [8., 0., 0., ..., 0., 0., 0.],
       [7., 0., 0., ..., 0., 0., 0.]], dtype=float32)
{% endhighlight %}

View the images of the data
```python
import random
# Select any random index from 1 to 60,000
i = random.randint(1,60000)
# Reshape and plot the image
plt.imshow( training[i,1:].reshape((28,28)) )
# Display the label of the image
label = training[i,0]
label
```
{% highlight text %}
6.0
{% endhighlight %}
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/image1.jpg" alt="">
The output is a 28x28 pixels image with a label of 6, indicating a shirt.
<br/>
View more images in a grid
```python
# Define the dimensions of the plot grid 
W_grid = 6
L_grid = 6

# subplot return the figure object and axes object
# We can use the axes object to plot specific figures at various locations
fig, axes = plt.subplots(L_grid, W_grid, figsize = (20,20))

# Flaten the 6 x 6 matrix into 36 array
axes = axes.ravel()

# Get the length of the training dataset
n_training = len(training)

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # Create evenly spaces variables 
    # Select a random number
    index = np.random.randint(0, n_training)
    # Read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 15)
    # Remove the axis showing the no. of pixels
    axes[i].axis('off')

# Spacing
plt.subplots_adjust(hspace=0.5)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/image2.jpg" alt="">

## 3) Training
Prepare the training and testing dataset. Since the image data in X_train and X_test is from 0 to 255 , we need to rescale this from 0 to 1 by performing a normalization.
<br/>
$${\text Normalization}= \frac{X - X_{\text min}}{X_{\text max} - X_{\text min}}= \frac{(X - 0)}{(255 - 0)}$$
<br/>
Therefore in order to do this we need to divide the X_train and X_test by 255
```python
X_train = training[:,1:]/255
y_train = training[:,0]
X_test = testing[:,1:]/255
y_test = testing[:,0]
```
Do a hold-out validation by spliting the data into 8:2 for training and testing
```python
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 5)
```
Reshape the data to be in a form of 28x28x1 (1 indicating a grayscale image), a form that the convolutional neural network will accept the data.
```python
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
```
View the shape of the data
```python
X_train.shape
```
{% highlight text %}
(48000, 28, 28, 1)
{% endhighlight %}
```python
X_test.shape
```
{% highlight text %}
(10000, 28, 28, 1)
{% endhighlight %}
```python
X_validate.shape
```
{% highlight text %}
(12000, 28, 28, 1)
{% endhighlight %}

Import libraries to perform the convolutional neural network
```python
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
```
Contructing the convolutional neural network layers
```python
cnn_model = Sequential()

# Try 32 fliters first then 64
cnn_model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))

# cnn_model.add(Conv2D(32,3, 3, activation='relu'))
# cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Flatten())
# Hidden layer
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
# Output layer
cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

epochs = 50
cnn_model.fit(X_train,
              y_train,
              batch_size = 512,
              nb_epoch = epochs,
              verbose = 1,
              validation_data = (X_validate, y_validate))
```
{% highlight text %}

{% endhighlight %}

## 4) Evaluating
```python
evaluation = cnn_model.evaluate(X_test, y_test)
# Print the accuracy which is the second element within the evaluation term
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```
{% highlight text %}
10000/10000 [==============================] - 4s 372us/step
Test Accuracy : 0.919
{% endhighlight %}


## 5) Improving the model
adding more feature detectors/filters
adding dropout



```python

```
{% highlight text %}

{% endhighlight %}
