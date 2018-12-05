---
title: Multiclass Classification with Fashion MNIST
date: 2018-10-16
tags: 
  - Neural Network
  - Classification
header:
  image: "/images/Fashion mnist/fashion.jpg"
  teaser: "/images/Fashion mnist/fashion.jpg"
excerpt: "Neural Network, Classification"
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
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep=',')
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
### View the shape of the dataframe
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
The shape conforms to the split of 60,000 training and 10,000 testing samples, with 785 columns
### Create training and testing arrays so that we can visualize the data
```python
training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype='float32')
```
### Viewing the arrays
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

### View the images of the data
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
<br/>
The output is a 28x28 pixels image with a label of 6, indicating a shirt.
<br/>
### View more images in a grid
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
Most of the classes are represented with an image.

## 3) Training
### Prepare the training and testing dataset. 
Since the image data in X_train and X_test is from 0 to 255 , we need to rescale this from 0 to 1 by performing a normalization.
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
### View the shape of the data
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

### Import libraries to perform the convolutional neural network
```python
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
```
### Contructing the convolutional neural network layers. 
By using the Sequential class with a linear stacks of layers consisting of Conv2D with relu activation, max pooling to downsample feature maps, flatten to a 1D tensor which is needed for processing in the dense layer, an intermediate dense layer with 32 units and a relu activation, a output dense layer of 10 units and a sigmoid activation as it is a multilabel classification problem. 
<br/>
For the compliation step, use an adam optimizer with learning rate of 0.001 which is a good default setting, the sparse_categorical_crossentropy as loss function since the labels are integers, an accuracy metrics to report on accuracy values.
```python
model = Sequential()
model.add(Conv2D(32,3, 3, input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
# Hidden layer
model.add(Dense(output_dim = 32, activation = 'relu'))
# Output layer
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Use sparse_categorical_crossentropy as loss function, since labels are integers
model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics =['accuracy'])
```

Train the model for 50 epochs, an iteration over all samples in the X_train and y_train, in mini batches of 512 samples and monitor the loss and accuracy on the validation data.
```python
# Run the model with the training set against the validation set
train_model = model.fit(X_train,
                        y_train,
                        batch_size = 512,
                        epochs = 50,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))
```
{% highlight text %}
Train on 48000 samples, validate on 12000 samples
Epoch 1/50
48000/48000 [==============================] - 5s 108us/step - loss: 0.8994 - acc: 0.6708 - val_loss: 0.5284 - val_acc: 0.8171
Epoch 2/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.4775 - acc: 0.8326 - val_loss: 0.4504 - val_acc: 0.8448
Epoch 3/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.4303 - acc: 0.8511 - val_loss: 0.4018 - val_acc: 0.8625
Epoch 4/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3981 - acc: 0.8617 - val_loss: 0.3814 - val_acc: 0.8702
Epoch 5/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3762 - acc: 0.8696 - val_loss: 0.3771 - val_acc: 0.8683
Epoch 6/50
48000/48000 [==============================] - 4s 90us/step - loss: 0.3582 - acc: 0.8759 - val_loss: 0.3554 - val_acc: 0.8792
Epoch 7/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3439 - acc: 0.8794 - val_loss: 0.3566 - val_acc: 0.8719
Epoch 8/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3357 - acc: 0.8832 - val_loss: 0.3322 - val_acc: 0.8857
Epoch 9/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3236 - acc: 0.8872 - val_loss: 0.3239 - val_acc: 0.8881
Epoch 10/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3111 - acc: 0.8914 - val_loss: 0.3217 - val_acc: 0.8858
Epoch 11/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.3060 - acc: 0.8929 - val_loss: 0.3188 - val_acc: 0.8887
Epoch 12/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2973 - acc: 0.8954 - val_loss: 0.3100 - val_acc: 0.8904
Epoch 13/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2899 - acc: 0.8971 - val_loss: 0.3047 - val_acc: 0.8938
Epoch 14/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2826 - acc: 0.9002 - val_loss: 0.2965 - val_acc: 0.8958
Epoch 15/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2769 - acc: 0.9020 - val_loss: 0.3172 - val_acc: 0.8878
Epoch 16/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2696 - acc: 0.9050 - val_loss: 0.2915 - val_acc: 0.8962
Epoch 17/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2676 - acc: 0.9052 - val_loss: 0.2888 - val_acc: 0.8984
Epoch 18/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2597 - acc: 0.9083 - val_loss: 0.2895 - val_acc: 0.9000
Epoch 19/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2574 - acc: 0.9092 - val_loss: 0.2786 - val_acc: 0.9016
Epoch 20/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2505 - acc: 0.9114 - val_loss: 0.2896 - val_acc: 0.8972
Epoch 21/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2439 - acc: 0.9149 - val_loss: 0.2856 - val_acc: 0.8987
Epoch 22/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2399 - acc: 0.9154 - val_loss: 0.2795 - val_acc: 0.8993
Epoch 23/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2380 - acc: 0.9157 - val_loss: 0.2811 - val_acc: 0.9008
Epoch 24/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2342 - acc: 0.9176 - val_loss: 0.2868 - val_acc: 0.8987
Epoch 25/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2289 - acc: 0.9192 - val_loss: 0.2733 - val_acc: 0.9052
Epoch 26/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2218 - acc: 0.9221 - val_loss: 0.2704 - val_acc: 0.9058
Epoch 27/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2205 - acc: 0.9222 - val_loss: 0.2743 - val_acc: 0.9039
Epoch 28/50
48000/48000 [==============================] - 4s 92us/step - loss: 0.2154 - acc: 0.9241 - val_loss: 0.2667 - val_acc: 0.9068
Epoch 29/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2137 - acc: 0.9250 - val_loss: 0.2666 - val_acc: 0.9065
Epoch 30/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2077 - acc: 0.9274 - val_loss: 0.2691 - val_acc: 0.9058
Epoch 31/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2059 - acc: 0.9270 - val_loss: 0.2734 - val_acc: 0.9017
Epoch 32/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2013 - acc: 0.9299 - val_loss: 0.2677 - val_acc: 0.9077
Epoch 33/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.2044 - acc: 0.9276 - val_loss: 0.2738 - val_acc: 0.9035
Epoch 34/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1940 - acc: 0.9323 - val_loss: 0.2629 - val_acc: 0.9066
Epoch 35/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1914 - acc: 0.9334 - val_loss: 0.2623 - val_acc: 0.9092
Epoch 36/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1881 - acc: 0.9348 - val_loss: 0.2737 - val_acc: 0.9034
Epoch 37/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1864 - acc: 0.9347 - val_loss: 0.2576 - val_acc: 0.9100
Epoch 38/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1813 - acc: 0.9367 - val_loss: 0.2606 - val_acc: 0.9099
Epoch 39/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1808 - acc: 0.9361 - val_loss: 0.2658 - val_acc: 0.9092
Epoch 40/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1761 - acc: 0.9373 - val_loss: 0.2680 - val_acc: 0.9083
Epoch 41/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1741 - acc: 0.9391 - val_loss: 0.2623 - val_acc: 0.9092
Epoch 42/50
48000/48000 [==============================] - 4s 90us/step - loss: 0.1678 - acc: 0.9421 - val_loss: 0.2613 - val_acc: 0.9075
Epoch 43/50
48000/48000 [==============================] - 5s 95us/step - loss: 0.1689 - acc: 0.9410 - val_loss: 0.2616 - val_acc: 0.9090
Epoch 44/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1632 - acc: 0.9433 - val_loss: 0.2623 - val_acc: 0.9105
Epoch 45/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1608 - acc: 0.9442 - val_loss: 0.2682 - val_acc: 0.9062
Epoch 46/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1560 - acc: 0.9461 - val_loss: 0.2777 - val_acc: 0.9077
Epoch 47/50
48000/48000 [==============================] - 4s 92us/step - loss: 0.1559 - acc: 0.9458 - val_loss: 0.2612 - val_acc: 0.9098
Epoch 48/50
48000/48000 [==============================] - 4s 92us/step - loss: 0.1524 - acc: 0.9470 - val_loss: 0.2644 - val_acc: 0.9097
Epoch 49/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1484 - acc: 0.9485 - val_loss: 0.2808 - val_acc: 0.9072
Epoch 50/50
48000/48000 [==============================] - 4s 91us/step - loss: 0.1493 - acc: 0.9475 - val_loss: 0.2648 - val_acc: 0.9078
{% endhighlight %}

Plot the training and validation accuracy and loss, using a history object defined in model.fit(), which is a dictionary containing data about everything that happened during training.
```python
hist = train_model.history
acc = hist['acc']
val_acc = hist['val_acc']
loss = hist['loss']
val_loss = hist['val_loss']
epochs = range(len(acc))
f, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(epochs, acc, 'g', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[1].plot(epochs, loss, 'g', label='Training loss')
ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/plot1.jpg" alt="">
These plot shows characteristic of overfitting. The training accuracy increases linearly over time until it reaches nearly 95%, whereas the validation accuracy stalls at 90%. The validation less reaches its minimum after about 19 epochs and then stalls, whereas the training loss keeps decreasing linearly until it reaches nearly 0.

### Evaluating on the test dataset
```python
evaluation = model.evaluate(X_test, y_test)
# Print the accuracy which is the second element within the evaluation term
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```
{% highlight text %}
10000/10000 [==============================] - 1s 123us/step
Test Accuracy : 0.913
{% endhighlight %}
The test accuracy reaches 91.3%. Try adding a dropout layer to mitigate the overfitting.

## 4) Add a dropout layer
Run using the same setting but with a Dropout layer with dropout rate of 0.25.
```python
model = Sequential()
model.add(Conv2D(32,3, 3, input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# Hidden layer
model.add(Dense(output_dim = 32, activation = 'relu'))
# Output layer
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Use sparse_categorical_crossentropy as loss function, since labels are integers
model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics =['accuracy'])
```

```python
# Run the model with the training set against the validation set
train_model = model.fit(X_train,
                        y_train,
                        batch_size = 512,
                        epochs = 50,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))
```
{% highlight text %}
Train on 48000 samples, validate on 12000 samples
Epoch 1/50
48000/48000 [==============================] - 7s 148us/step - loss: 0.9567 - acc: 0.6240 - val_loss: 0.5037 - val_acc: 0.8137
Epoch 2/50
48000/48000 [==============================] - 6s 119us/step - loss: 0.4653 - acc: 0.8357 - val_loss: 0.4104 - val_acc: 0.8578
Epoch 3/50
48000/48000 [==============================] - 6s 122us/step - loss: 0.4130 - acc: 0.8551 - val_loss: 0.3821 - val_acc: 0.8674
Epoch 4/50
48000/48000 [==============================] - 6s 122us/step - loss: 0.3878 - acc: 0.8634 - val_loss: 0.3568 - val_acc: 0.8773
Epoch 5/50
48000/48000 [==============================] - 6s 117us/step - loss: 0.3639 - acc: 0.8709 - val_loss: 0.3450 - val_acc: 0.8815
Epoch 6/50
48000/48000 [==============================] - 6s 119us/step - loss: 0.3441 - acc: 0.8789 - val_loss: 0.3346 - val_acc: 0.8860
Epoch 7/50
48000/48000 [==============================] - 6s 118us/step - loss: 0.3288 - acc: 0.8860 - val_loss: 0.3196 - val_acc: 0.8904
Epoch 8/50
48000/48000 [==============================] - 6s 126us/step - loss: 0.3201 - acc: 0.8881 - val_loss: 0.3083 - val_acc: 0.8960
Epoch 9/50
48000/48000 [==============================] - 6s 119us/step - loss: 0.3122 - acc: 0.8904 - val_loss: 0.3125 - val_acc: 0.8898
Epoch 10/50
48000/48000 [==============================] - 6s 116us/step - loss: 0.3039 - acc: 0.8934 - val_loss: 0.3064 - val_acc: 0.8928
Epoch 11/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2966 - acc: 0.8957 - val_loss: 0.3007 - val_acc: 0.8935
Epoch 12/50
48000/48000 [==============================] - 5s 112us/step - loss: 0.2944 - acc: 0.8962 - val_loss: 0.2866 - val_acc: 0.9023
Epoch 13/50
48000/48000 [==============================] - 5s 112us/step - loss: 0.2841 - acc: 0.8992 - val_loss: 0.2881 - val_acc: 0.8993
Epoch 14/50
48000/48000 [==============================] - 6s 117us/step - loss: 0.2803 - acc: 0.9014 - val_loss: 0.2955 - val_acc: 0.8936
Epoch 15/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2776 - acc: 0.9008 - val_loss: 0.2862 - val_acc: 0.9007
Epoch 16/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2708 - acc: 0.9033 - val_loss: 0.2792 - val_acc: 0.9030
Epoch 17/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2706 - acc: 0.9031 - val_loss: 0.2711 - val_acc: 0.9063
Epoch 18/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2627 - acc: 0.9062 - val_loss: 0.2683 - val_acc: 0.9073
Epoch 19/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2586 - acc: 0.9074 - val_loss: 0.2694 - val_acc: 0.9048
Epoch 20/50
48000/48000 [==============================] - 6s 117us/step - loss: 0.2551 - acc: 0.9085 - val_loss: 0.2830 - val_acc: 0.8981
Epoch 21/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2570 - acc: 0.9071 - val_loss: 0.2762 - val_acc: 0.9025
Epoch 22/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2491 - acc: 0.9102 - val_loss: 0.2631 - val_acc: 0.9095
Epoch 23/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2473 - acc: 0.9121 - val_loss: 0.2648 - val_acc: 0.9081
Epoch 24/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2459 - acc: 0.9122 - val_loss: 0.2609 - val_acc: 0.9095
Epoch 25/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2419 - acc: 0.9136 - val_loss: 0.2635 - val_acc: 0.9050
Epoch 26/50
48000/48000 [==============================] - 6s 118us/step - loss: 0.2432 - acc: 0.9129 - val_loss: 0.2596 - val_acc: 0.9091
Epoch 27/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2384 - acc: 0.9141 - val_loss: 0.2657 - val_acc: 0.9072
Epoch 28/50
48000/48000 [==============================] - 6s 122us/step - loss: 0.2356 - acc: 0.9155 - val_loss: 0.2608 - val_acc: 0.9076
Epoch 29/50
48000/48000 [==============================] - 6s 126us/step - loss: 0.2325 - acc: 0.9155 - val_loss: 0.2555 - val_acc: 0.9098
Epoch 30/50
48000/48000 [==============================] - 6s 122us/step - loss: 0.2286 - acc: 0.9180 - val_loss: 0.2583 - val_acc: 0.9091
Epoch 31/50
48000/48000 [==============================] - 6s 120us/step - loss: 0.2278 - acc: 0.9176 - val_loss: 0.2574 - val_acc: 0.9077
Epoch 32/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2258 - acc: 0.9182 - val_loss: 0.2566 - val_acc: 0.9115
Epoch 33/50
48000/48000 [==============================] - 5s 112us/step - loss: 0.2214 - acc: 0.9191 - val_loss: 0.2579 - val_acc: 0.9098
Epoch 34/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2186 - acc: 0.9200 - val_loss: 0.2580 - val_acc: 0.9088
Epoch 35/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2158 - acc: 0.9224 - val_loss: 0.2476 - val_acc: 0.9136
Epoch 36/50
48000/48000 [==============================] - 6s 118us/step - loss: 0.2102 - acc: 0.9235 - val_loss: 0.2562 - val_acc: 0.9091
Epoch 37/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2154 - acc: 0.9216 - val_loss: 0.2510 - val_acc: 0.9140
Epoch 38/50
48000/48000 [==============================] - 6s 115us/step - loss: 0.2093 - acc: 0.9245 - val_loss: 0.2439 - val_acc: 0.9150
Epoch 39/50
48000/48000 [==============================] - 6s 120us/step - loss: 0.2074 - acc: 0.9248 - val_loss: 0.2529 - val_acc: 0.9126
Epoch 40/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2039 - acc: 0.9255 - val_loss: 0.2445 - val_acc: 0.9143
Epoch 41/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2027 - acc: 0.9255 - val_loss: 0.2505 - val_acc: 0.9133
Epoch 42/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2005 - acc: 0.9267 - val_loss: 0.2504 - val_acc: 0.9102
Epoch 43/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.2040 - acc: 0.9259 - val_loss: 0.2512 - val_acc: 0.9135
Epoch 44/50
48000/48000 [==============================] - 5s 113us/step - loss: 0.1956 - acc: 0.9286 - val_loss: 0.2475 - val_acc: 0.9142
Epoch 45/50
48000/48000 [==============================] - 5s 113us/step - loss: 0.1950 - acc: 0.9289 - val_loss: 0.2509 - val_acc: 0.9115
Epoch 46/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.1909 - acc: 0.9304 - val_loss: 0.2431 - val_acc: 0.9139
Epoch 47/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.1928 - acc: 0.9299 - val_loss: 0.2505 - val_acc: 0.9112
Epoch 48/50
48000/48000 [==============================] - 5s 114us/step - loss: 0.1882 - acc: 0.9322 - val_loss: 0.2493 - val_acc: 0.9142
Epoch 49/50
48000/48000 [==============================] - 5s 113us/step - loss: 0.1872 - acc: 0.9319 - val_loss: 0.2516 - val_acc: 0.9118
Epoch 50/50
48000/48000 [==============================] - 5s 113us/step - loss: 0.1887 - acc: 0.9311 - val_loss: 0.2440 - val_acc: 0.9150
{% endhighlight %}

```python
# Plot the training and validation accuracy and loss, from the training history.
hist = train_model.history
acc = hist['acc']
val_acc = hist['val_acc']
loss = hist['loss']
val_loss = hist['val_loss']
epochs = range(len(acc))
f, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(epochs, acc, 'g', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[1].plot(epochs, loss, 'g', label='Training loss')
ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/plot2.jpg" alt="">
The model reach a validation accuracy of about 91.5%, which is much better than the previous model. With the dropout layer, the model is also much better at generalizing.
<br/>

### Re-evaluating the test dataset on the new model
```python
evaluation = model.evaluate(X_test, y_test)
# Print the accuracy which is the second element within the evaluation term
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```
{% highlight text %}
10000/10000 [==============================] - 1s 125us/step
Test Accuracy : 0.915
{% endhighlight %}
The test accuracy reaches 91.5% which happens to be the same as the validation accuracy and is also performing slightly better than the previous model.

## 5) Increasing to 64 filters
Try improving the accuracy by increasing the no. of filters/kernels to 64 with the same settings and the dropout layer.
```python
model = Sequential()
model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# Hidden layer
model.add(Dense(output_dim = 32, activation = 'relu'))
# Output layer
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Use sparse_categorical_crossentropy as loss function, since labels are integers
model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics =['accuracy'])
```

```python
# Run the model with the training set against the validation set
train_model = model.fit(X_train,
                        y_train,
                        batch_size = 512,
                        epochs = 50,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))
```
{% highlight text %}
Train on 48000 samples, validate on 12000 samples
Epoch 1/50
48000/48000 [==============================] - 10s 215us/step - loss: 0.7871 - acc: 0.7383 - val_loss: 0.4663 - val_acc: 0.8326
Epoch 2/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.4387 - acc: 0.8465 - val_loss: 0.3970 - val_acc: 0.8668
Epoch 3/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.3822 - acc: 0.8669 - val_loss: 0.3581 - val_acc: 0.8784
Epoch 4/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.3532 - acc: 0.8767 - val_loss: 0.3380 - val_acc: 0.8824
Epoch 5/50
48000/48000 [==============================] - 10s 202us/step - loss: 0.3278 - acc: 0.8852 - val_loss: 0.3246 - val_acc: 0.8890
Epoch 6/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.3193 - acc: 0.8886 - val_loss: 0.3237 - val_acc: 0.8865
Epoch 7/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.2990 - acc: 0.8938 - val_loss: 0.3087 - val_acc: 0.8929
Epoch 8/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.2891 - acc: 0.8972 - val_loss: 0.2937 - val_acc: 0.8985
Epoch 9/50
48000/48000 [==============================] - 10s 202us/step - loss: 0.2793 - acc: 0.9012 - val_loss: 0.2833 - val_acc: 0.9014
Epoch 10/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2716 - acc: 0.9040 - val_loss: 0.2844 - val_acc: 0.9017
Epoch 11/50
48000/48000 [==============================] - 10s 208us/step - loss: 0.2657 - acc: 0.9049 - val_loss: 0.2740 - val_acc: 0.9054
Epoch 12/50
48000/48000 [==============================] - 10s 209us/step - loss: 0.2552 - acc: 0.9097 - val_loss: 0.2727 - val_acc: 0.9041
Epoch 13/50
48000/48000 [==============================] - 10s 203us/step - loss: 0.2511 - acc: 0.9102 - val_loss: 0.2677 - val_acc: 0.9040
Epoch 14/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2442 - acc: 0.9132 - val_loss: 0.2674 - val_acc: 0.9057
Epoch 15/50
48000/48000 [==============================] - 10s 202us/step - loss: 0.2376 - acc: 0.9150 - val_loss: 0.2687 - val_acc: 0.9051
Epoch 16/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2338 - acc: 0.9160 - val_loss: 0.2605 - val_acc: 0.9087
Epoch 17/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2250 - acc: 0.9196 - val_loss: 0.2562 - val_acc: 0.9106
Epoch 18/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.2254 - acc: 0.9198 - val_loss: 0.2538 - val_acc: 0.9122
Epoch 19/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2186 - acc: 0.9226 - val_loss: 0.2653 - val_acc: 0.9073
Epoch 20/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.2152 - acc: 0.9219 - val_loss: 0.2586 - val_acc: 0.9096
Epoch 21/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.2079 - acc: 0.9254 - val_loss: 0.2470 - val_acc: 0.9138
Epoch 22/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.2049 - acc: 0.9264 - val_loss: 0.2535 - val_acc: 0.9115
Epoch 23/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1993 - acc: 0.9289 - val_loss: 0.2402 - val_acc: 0.9173
Epoch 24/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.1933 - acc: 0.9304 - val_loss: 0.2393 - val_acc: 0.9171
Epoch 25/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.1911 - acc: 0.9311 - val_loss: 0.2435 - val_acc: 0.9157
Epoch 26/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1883 - acc: 0.9320 - val_loss: 0.2568 - val_acc: 0.9087
Epoch 27/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.1908 - acc: 0.9310 - val_loss: 0.2396 - val_acc: 0.9169
Epoch 28/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1853 - acc: 0.9330 - val_loss: 0.2564 - val_acc: 0.9120
Epoch 29/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.1801 - acc: 0.9350 - val_loss: 0.2440 - val_acc: 0.9155
Epoch 30/50
48000/48000 [==============================] - 10s 202us/step - loss: 0.1783 - acc: 0.9357 - val_loss: 0.2464 - val_acc: 0.9148
Epoch 31/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1743 - acc: 0.9369 - val_loss: 0.2481 - val_acc: 0.9155
Epoch 32/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1708 - acc: 0.9382 - val_loss: 0.2465 - val_acc: 0.9147
Epoch 33/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1703 - acc: 0.9383 - val_loss: 0.2431 - val_acc: 0.9157
Epoch 34/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1666 - acc: 0.9395 - val_loss: 0.2452 - val_acc: 0.9145
Epoch 35/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1600 - acc: 0.9412 - val_loss: 0.2449 - val_acc: 0.9171
Epoch 36/50
48000/48000 [==============================] - 10s 202us/step - loss: 0.1595 - acc: 0.9420 - val_loss: 0.2376 - val_acc: 0.9172
Epoch 37/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1558 - acc: 0.9440 - val_loss: 0.2395 - val_acc: 0.9177
Epoch 38/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1567 - acc: 0.9433 - val_loss: 0.2516 - val_acc: 0.9149
Epoch 39/50
48000/48000 [==============================] - 10s 200us/step - loss: 0.1528 - acc: 0.9441 - val_loss: 0.2382 - val_acc: 0.9190
Epoch 40/50
48000/48000 [==============================] - 10s 201us/step - loss: 0.1473 - acc: 0.9465 - val_loss: 0.2410 - val_acc: 0.9191
Epoch 41/50
48000/48000 [==============================] - 10s 203us/step - loss: 0.1495 - acc: 0.9456 - val_loss: 0.2359 - val_acc: 0.9196
Epoch 42/50
48000/48000 [==============================] - 10s 203us/step - loss: 0.1435 - acc: 0.9477 - val_loss: 0.2393 - val_acc: 0.9195
Epoch 43/50
48000/48000 [==============================] - 10s 205us/step - loss: 0.1417 - acc: 0.9489 - val_loss: 0.2403 - val_acc: 0.9185
Epoch 44/50
48000/48000 [==============================] - 10s 205us/step - loss: 0.1375 - acc: 0.9506 - val_loss: 0.2434 - val_acc: 0.9168
Epoch 45/50
48000/48000 [==============================] - 10s 206us/step - loss: 0.1394 - acc: 0.9490 - val_loss: 0.2453 - val_acc: 0.9181
Epoch 46/50
48000/48000 [==============================] - 10s 207us/step - loss: 0.1363 - acc: 0.9504 - val_loss: 0.2592 - val_acc: 0.9118
Epoch 47/50
48000/48000 [==============================] - 10s 207us/step - loss: 0.1377 - acc: 0.9501 - val_loss: 0.2542 - val_acc: 0.9152
Epoch 48/50
48000/48000 [==============================] - 10s 208us/step - loss: 0.1341 - acc: 0.9515 - val_loss: 0.2409 - val_acc: 0.9214
Epoch 49/50
48000/48000 [==============================] - 10s 207us/step - loss: 0.1311 - acc: 0.9512 - val_loss: 0.2615 - val_acc: 0.9158
Epoch 50/50
48000/48000 [==============================] - 10s 208us/step - loss: 0.1351 - acc: 0.9505 - val_loss: 0.2484 - val_acc: 0.9192
{% endhighlight %}

```python
# Plot the training and validation accuracy and loss, from the training history.
hist = train_model.history
acc = hist['acc']
val_acc = hist['val_acc']
loss = hist['loss']
val_loss = hist['val_loss']
epochs = range(len(acc))
f, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(epochs, acc, 'g', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[1].plot(epochs, loss, 'g', label='Training loss')
ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/plot3.jpg" alt="">
These plot shows characteristic of overfitting. The model begins to overfit after about 15 epochs and the validation accuracy and loss begins to stall.

### Evaluating against the test dataset
```python
evaluation = model.evaluate(X_test, y_test)
# Print the accuracy which is the second element within the evaluation term
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```
{% highlight text %}
10000/10000 [==============================] - 2s 153us/step
Test Accuracy : 0.919
{% endhighlight %}
The test accuracy reaches 91.9%

## 6) Increasing to 128 filters with extra layers
Try to further improve the accuracy by increasing the no. of filters/kernels to 128 with the same settings and a stack of alternated Conv2D and MaxPooling2D layers.
```python
model = Sequential()
model.add(Conv2D(128,3, 3, input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# Hidden layer
model.add(Dense(output_dim = 32, activation = 'relu'))
# Output layer
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Use sparse_categorical_crossentropy as loss function, since labels are integers
model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
```

```python
# Run the model with the training set against the validation set
train_model = model.fit(X_train,
                        y_train,
                        batch_size = 512,
                        epochs = 50,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))
```
{% highlight text %}
Train on 48000 samples, validate on 12000 samples
Epoch 1/50
48000/48000 [==============================] - 25s 522us/step - loss: 0.9996 - acc: 0.6433 - val_loss: 0.5808 - val_acc: 0.7863
Epoch 2/50
48000/48000 [==============================] - 24s 500us/step - loss: 0.5327 - acc: 0.8065 - val_loss: 0.4416 - val_acc: 0.8430
Epoch 3/50
48000/48000 [==============================] - 24s 506us/step - loss: 0.4439 - acc: 0.8417 - val_loss: 0.3930 - val_acc: 0.8624
Epoch 4/50
48000/48000 [==============================] - 25s 513us/step - loss: 0.4089 - acc: 0.8537 - val_loss: 0.3668 - val_acc: 0.8711
Epoch 5/50
48000/48000 [==============================] - 25s 511us/step - loss: 0.3871 - acc: 0.8608 - val_loss: 0.3500 - val_acc: 0.8792
Epoch 6/50
48000/48000 [==============================] - 25s 511us/step - loss: 0.3584 - acc: 0.8728 - val_loss: 0.3404 - val_acc: 0.8821
Epoch 7/50
48000/48000 [==============================] - 25s 511us/step - loss: 0.3459 - acc: 0.8768 - val_loss: 0.3150 - val_acc: 0.8860
Epoch 8/50
48000/48000 [==============================] - 25s 511us/step - loss: 0.3316 - acc: 0.8814 - val_loss: 0.3044 - val_acc: 0.8924
Epoch 9/50
48000/48000 [==============================] - 24s 510us/step - loss: 0.3203 - acc: 0.8855 - val_loss: 0.2971 - val_acc: 0.8943
Epoch 10/50
48000/48000 [==============================] - 24s 510us/step - loss: 0.3074 - acc: 0.8901 - val_loss: 0.2972 - val_acc: 0.8928
Epoch 11/50
48000/48000 [==============================] - 24s 510us/step - loss: 0.3005 - acc: 0.8926 - val_loss: 0.2895 - val_acc: 0.8986
Epoch 12/50
48000/48000 [==============================] - 25s 516us/step - loss: 0.2900 - acc: 0.8959 - val_loss: 0.2733 - val_acc: 0.9034
Epoch 13/50
48000/48000 [==============================] - 25s 521us/step - loss: 0.2804 - acc: 0.8975 - val_loss: 0.2869 - val_acc: 0.8984
Epoch 14/50
48000/48000 [==============================] - 25s 514us/step - loss: 0.2763 - acc: 0.9017 - val_loss: 0.2776 - val_acc: 0.9002
Epoch 15/50
48000/48000 [==============================] - 25s 516us/step - loss: 0.2667 - acc: 0.9043 - val_loss: 0.2608 - val_acc: 0.9063
Epoch 16/50
48000/48000 [==============================] - 24s 509us/step - loss: 0.2620 - acc: 0.9061 - val_loss: 0.2518 - val_acc: 0.9102
Epoch 17/50
48000/48000 [==============================] - 24s 507us/step - loss: 0.2567 - acc: 0.9060 - val_loss: 0.2728 - val_acc: 0.9015
Epoch 18/50
48000/48000 [==============================] - 24s 506us/step - loss: 0.2485 - acc: 0.9104 - val_loss: 0.2468 - val_acc: 0.9114
Epoch 19/50
48000/48000 [==============================] - 24s 507us/step - loss: 0.2417 - acc: 0.9129 - val_loss: 0.2469 - val_acc: 0.9102
Epoch 20/50
48000/48000 [==============================] - 24s 506us/step - loss: 0.2367 - acc: 0.9150 - val_loss: 0.2440 - val_acc: 0.9116
Epoch 21/50
48000/48000 [==============================] - 24s 502us/step - loss: 0.2338 - acc: 0.9152 - val_loss: 0.2476 - val_acc: 0.9084
Epoch 22/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2305 - acc: 0.9169 - val_loss: 0.2522 - val_acc: 0.9062
Epoch 23/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2220 - acc: 0.9184 - val_loss: 0.2427 - val_acc: 0.9123
Epoch 24/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2179 - acc: 0.9209 - val_loss: 0.2393 - val_acc: 0.9135
Epoch 25/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2093 - acc: 0.9240 - val_loss: 0.2313 - val_acc: 0.9144
Epoch 26/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2061 - acc: 0.9251 - val_loss: 0.2310 - val_acc: 0.9154
Epoch 27/50
48000/48000 [==============================] - 24s 495us/step - loss: 0.2046 - acc: 0.9249 - val_loss: 0.2336 - val_acc: 0.9137
Epoch 28/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.2016 - acc: 0.9273 - val_loss: 0.2361 - val_acc: 0.9126
Epoch 29/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1979 - acc: 0.9279 - val_loss: 0.2298 - val_acc: 0.9140
Epoch 30/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1906 - acc: 0.9305 - val_loss: 0.2279 - val_acc: 0.9182
Epoch 31/50
48000/48000 [==============================] - 24s 495us/step - loss: 0.1905 - acc: 0.9296 - val_loss: 0.2426 - val_acc: 0.9133
Epoch 32/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1835 - acc: 0.9334 - val_loss: 0.2279 - val_acc: 0.9187
Epoch 33/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1819 - acc: 0.9328 - val_loss: 0.2242 - val_acc: 0.9196
Epoch 34/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1783 - acc: 0.9348 - val_loss: 0.2256 - val_acc: 0.9192
Epoch 35/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1772 - acc: 0.9340 - val_loss: 0.2237 - val_acc: 0.9198
Epoch 36/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1732 - acc: 0.9363 - val_loss: 0.2242 - val_acc: 0.9180
Epoch 37/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1669 - acc: 0.9394 - val_loss: 0.2273 - val_acc: 0.9167
Epoch 38/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1660 - acc: 0.9390 - val_loss: 0.2260 - val_acc: 0.9163
Epoch 39/50
48000/48000 [==============================] - 24s 495us/step - loss: 0.1607 - acc: 0.9402 - val_loss: 0.2272 - val_acc: 0.9170
Epoch 40/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1612 - acc: 0.9398 - val_loss: 0.2231 - val_acc: 0.9186
Epoch 41/50
48000/48000 [==============================] - 24s 494us/step - loss: 0.1588 - acc: 0.9415 - val_loss: 0.2196 - val_acc: 0.9208
Epoch 42/50
48000/48000 [==============================] - 24s 495us/step - loss: 0.1562 - acc: 0.9410 - val_loss: 0.2283 - val_acc: 0.9211
Epoch 43/50
48000/48000 [==============================] - 24s 498us/step - loss: 0.1485 - acc: 0.9452 - val_loss: 0.2347 - val_acc: 0.9184
Epoch 44/50
48000/48000 [==============================] - 24s 506us/step - loss: 0.1502 - acc: 0.9448 - val_loss: 0.2375 - val_acc: 0.9141
Epoch 45/50
48000/48000 [==============================] - 24s 498us/step - loss: 0.1481 - acc: 0.9450 - val_loss: 0.2237 - val_acc: 0.9207
Epoch 46/50
48000/48000 [==============================] - 24s 502us/step - loss: 0.1422 - acc: 0.9482 - val_loss: 0.2284 - val_acc: 0.9197
Epoch 47/50
48000/48000 [==============================] - 24s 499us/step - loss: 0.1436 - acc: 0.9474 - val_loss: 0.2225 - val_acc: 0.9198
Epoch 48/50
48000/48000 [==============================] - 24s 499us/step - loss: 0.1381 - acc: 0.9492 - val_loss: 0.2240 - val_acc: 0.9203
Epoch 49/50
48000/48000 [==============================] - 24s 499us/step - loss: 0.1396 - acc: 0.9482 - val_loss: 0.2207 - val_acc: 0.9237
Epoch 50/50
48000/48000 [==============================] - 25s 511us/step - loss: 0.1368 - acc: 0.9490 - val_loss: 0.2270 - val_acc: 0.9208
{% endhighlight %}

```python
# Plot the training and validation accuracy and loss, from the training history.
hist = train_model.history
acc = hist['acc']
val_acc = hist['val_acc']
loss = hist['loss']
val_loss = hist['val_loss']
epochs = range(len(acc))
f, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(epochs, acc, 'g', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[1].plot(epochs, loss, 'g', label='Training loss')
ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/plot4.jpg" alt="">
There are signs of overfitting from the model after about 20 epochs and the validation accuracy and loss begins to stall. 

### Evaluating against the test dataset
```python
# Re-evaluate the test prediction accuracy with the new model
evaluation = model.evaluate(X_test, y_test)
# Print the accuracy which is the second element within the evaluation term
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
```
{% highlight text %}
10000/10000 [==============================] - 3s 274us/step
Test Accuracy : 0.926
{% endhighlight %}
The test accuracy reaches 92.6%

## 7) Reviewing on the model
### Get the predictions for the test data
```python
predicted_classes = model.predict_classes(X_test)
```

### Display the prediction against true class on a grid
```python
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/image3.jpg" alt="">
We could see the some classes are predicted wrongly, like on grid (2,1) and (1,4). The true class is a pullover, however the model predicted it to be a shirt and coat respectively. Indeed, it's hard to distinguish between those 3 classes.

### Viewing the confusion matrix on a heatmap
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, cmap='pink', annot=True)
# Sum the diagonal element to get the total true correct values
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Fashion mnist/heatmap.jpg" alt="">
Seems like class 6, pullover had the most misclassification.

### View the classification report of all classes
```python
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))
```
{% highlight text %}
             precision    recall  f1-score   support

    Class 0       0.89      0.85      0.87      1000
    Class 1       0.99      0.99      0.99      1000
    Class 2       0.89      0.90      0.89      1000
    Class 3       0.95      0.91      0.93      1000
    Class 4       0.87      0.92      0.89      1000
    Class 5       0.98      0.99      0.98      1000
    Class 6       0.79      0.80      0.80      1000
    Class 7       0.97      0.94      0.95      1000
    Class 8       0.98      0.99      0.99      1000
    Class 9       0.95      0.98      0.97      1000

avg / total       0.93      0.93      0.93     10000
{% endhighlight %}

Overall, Class 6 had the lowest f1 score, which is the harmonic average of the precision and recall.
