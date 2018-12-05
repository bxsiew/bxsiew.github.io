---
title: Generating Text with RNN Long Short-Term Memory
date: 2018-10-20
tags: 
  - Neural Network
  - Text Analytics
header:
  image: "/images/Generating Text/book.jpg"
  teaser: "/images/Generating Text/book.jpg"
excerpt: "Neural Network, Text Analytics"
---

The aim is to develop a model generating text, character by character using LSTM recurrent neural networks with Keras. 
<br/>

The dataset used to create the generative model is taken from Project Gutenberg, a site to get access to free books that are no longer protected by copyright. We will be using the text from Alice's Adventures in Wonderland by Lewis Carroll to train the model and also be running on GPU to speed up the computation.
<img src="{{ site.url }}{{ site.baseurl }}/images/Generating Text/Alice colour.png" alt="">
## 1) Setup
### Load packages
```python
import numpy
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
```

### Load & Viewing the Data
Load the text file for the book and convert all of the characters to lowercase to reduce the number of vocabulary that the model must learn.
```python
# load ascii text and convert to lowercase
text = open("wonderland.txt").read()
text = text.lower()
```

### Viewing the first 1000 characters
```python
text[:1000]
```
{% highlight text %}
"ï»¿chapter i. down the rabbit-hole\n\nalice was beginning to get very tired of sitting by her 
sister on the\nbank, and of having nothing to do: once or twice she had peeped into the\nbook 
her sister was reading, but it had no pictures or conversations in\nit, 'and what is the use of 
a book,' thought alice 'without pictures or\nconversations?'\n\nso she was considering in her 
own mind (as well as she could, for the\nhot day made her feel very sleepy and stupid), whether 
the pleasure\nof making a daisy-chain would be worth the trouble of getting up and\npicking the 
daisies, when suddenly a white rabbit with pink eyes ran\nclose by her.\n\nthere was nothing so 
very remarkable in that; nor did alice think it so\nvery much out of the way to hear the rabbit 
say to itself, 'oh dear!\noh dear! i shall be late!' (when she thought it over afterwards, it\noccurred 
to her that she ought to have wondered at this, but at the time\nit all seemed quite natural); 
but when the rabbit actually took a watch\nout of its wais"
{% endhighlight %} 

## 2) Data Preprocessing
### Create mapping of unique characters to integers and a reverse mapping
Convert the unique characters to integers, by first creating a set of all distinct characters from the book and then create a mapping of each character to a unique integer. Also create a reverse mapping that can be use to convert the integers back to characters so that we can understand the predictions.
```python
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
```
### Display the unique vocabulary
```python
print(chars)
```
{% highlight text %}
['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '»', '¿', 'ï']
{% endhighlight %} 
We can see that there may be some characters that could be removed to reduce the vocabulary further and improve the modeling process.

### Summarize the loaded data
```python
n_chars = len(text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
```
{% highlight text %}
Total Characters:  144345
Total Vocab:  46
{% endhighlight %} 
The book have a total of 144,345 characters and when converted to lowercase there are 46 distinct characters in the vocabulary.

### Prepare the dataset of input to output pairs encoded as integers
Split the book's text into subsequences with a fixed length of 100 characters, an arbitrary length. The for loop is used to iterate over the entire characters of the text and create individual training pattern comprised of 100 time steps of one character(dataX) followed by one character output(dataY). When creating these sequences, the sequence length is slided along the whole book one character at a time, allowing each character to be learned from the 100 characters that preceded it.
<br/>
Example, a sequence length of 5 with the text "alice in wonderland", the first 5 training patterns(not mapped as integers for simplicity) would be as follows:
<br/>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>dataX</th>
      <th>dataY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>[a,l,i,c,e]</td>
      <td>[ ]</td>
    </tr>
    <tr>
      <td>[l,i,c,e, ]</td>
      <td>[i]</td>
    </tr>
    <tr>
      <td>[i,c,e, ,i]</td>
      <td>[n]</td>
    </tr>
    <tr>
      <td>[c,e, ,i,n]</td>
      <td>[ ]</td>
    </tr>
    <tr>
      <td>[e, ,i,n, ]</td>
      <td>[w]</td>
    </tr>
  </tbody>
</table>

```python
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
```
{% highlight text %}
Total Patterns:  144245
{% endhighlight %} 
The total number of patterns is 144245, which makes sense as excluding the first 100 characters, we have one training pattern each to predict the remaining characters.

### Reshape into the form [samples, time steps, features]
Transform the list of input sequences into the form expected by an LSTM network.
```python
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
```
### Normalization
Rescale the integers to the range 0 to 1, which make the patterns easier to be learn by the LSTM network that uses the sigmoid activation function by default.
```python
X = X / float(n_vocab)
```
### One hot encode
Convert the output variables(single characters) into a one hot encoding, so that the network can predict the probability of each of the 46 different characters in the vocabulary. The output will be a sparse vector with a length of 46.
```python
y = np_utils.to_categorical(dataY)
```

## 3) Training the model
Develop a LSTM network to learn sequences of characters from Alice in Wonderland and generate new sequences of characters.
### Define the LSTM model
First using the Sequential class with a stacks of layers consisting of a single hidden LSTM layer with 256 memory units, followed by a dropout layer with dropout rate of 20% and a output dense layer with a softmax activation function to output a probability prediction for each of the 46 characters between 0 and 1.
<br/>
For the compliation step, since this is really a classification problem with 46 classes and as such is defined as optimizing the log loss(cross entropy), followed by the adam optimization for speed.
```python
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

### Fit the model
Train the model for 20 epochs, an iteration over all samples in the X and y, in batches of 128 patterns.
```python
model.fit(X, y, epochs=20, batch_size=128)
```
{% highlight text %}
Epoch 1/20
144245/144245 [==============================] - 364s 3ms/step - loss: 2.9702
Epoch 2/20
144245/144245 [==============================] - 352s 2ms/step - loss: 2.7547
Epoch 3/20
144245/144245 [==============================] - 349s 2ms/step - loss: 2.6528
Epoch 4/20
144245/144245 [==============================] - 361s 3ms/step - loss: 2.5789
Epoch 5/20
144245/144245 [==============================] - 332s 2ms/step - loss: 2.5194
Epoch 6/20
144245/144245 [==============================] - 345s 2ms/step - loss: 2.4629
Epoch 7/20
144245/144245 [==============================] - 341s 2ms/step - loss: 2.4090
Epoch 8/20
144245/144245 [==============================] - 354s 2ms/step - loss: 2.3580
Epoch 9/20
144245/144245 [==============================] - 338s 2ms/step - loss: 2.3105
Epoch 10/20
144245/144245 [==============================] - 343s 2ms/step - loss: 2.2679
Epoch 11/20
144245/144245 [==============================] - 338s 2ms/step - loss: 2.2376
Epoch 12/20
144245/144245 [==============================] - 393s 3ms/step - loss: 2.1909
Epoch 13/20
144245/144245 [==============================] - 389s 3ms/step - loss: 2.1583
Epoch 14/20
144245/144245 [==============================] - 401s 3ms/step - loss: 2.1302
Epoch 15/20
144245/144245 [==============================] - 394s 3ms/step - loss: 2.0903
Epoch 16/20
144245/144245 [==============================] - 400s 3ms/step - loss: 2.0592
Epoch 17/20
144245/144245 [==============================] - 337s 2ms/step - loss: 2.0259
Epoch 18/20
144245/144245 [==============================] - 333s 2ms/step - loss: 1.9953
Epoch 19/20
144245/144245 [==============================] - 337s 2ms/step - loss: 1.9669
Epoch 20/20
144245/144245 [==============================] - 405s 3ms/step - loss: 1.9413
{% endhighlight %} 

### Pick a random seed sequence
In order to make predictions, first we start off with a seed sequence(a random input pattern) as input.
```python
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
```
{% highlight text %}
Seed:
"  pocket?' he went on, turning to alice.

'only a thimble,' said alice sadly.

'hand it over here,' s "
{% endhighlight %} 

### Generate characters
With a seed sequence as input, generate the next character then update the seed sequence to add the generated character on the end and trim off the first character. This process is then repeated for as long as we want to predict new characters (a sequence of 1,000 characters in length, an arbitrary number).
```python
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
```
{% highlight text %}
aid the konk, and een tees to toe toiee to the thitg sae oo the that was io an anc oo the kooer 
sote  the had never her aelin the woide anice was soe oittle thing to be so the kooer oo tho 
thet was the catt with sote toene at hnr aadkne the woide of the toees was oo tie tiat was in a 
conner of the sooe, and she whotg taemed thre toon hn soeh a aroleus to the woode of the toees 
was a lontte oo tho what she was not abrit at in saading ao the could, the was aolinger deaa so 
the thitg thete the was no the toiee of the coorersing at the could  and the borltuse so sele 
the wiilg rabdit would the was no the thit har and the boorerseng at the could, and the thought 
it was soeer the soeer saadit  and the thitghr to the toiee the harten  and the thought it was 
soeer the sooe of the tooes  and then the was not a conner of the saale, and the whot har aedin 
to the kooe oa thing the had so tee at hlr hanee an the could  aadin to the korer of thing the 
sabe th the korer oa thite the soies tay oo 
Done.
{% endhighlight %} 
Note some observations about the generate text:
* It generally conforms to the line format observed in the original text.
* The characters are separated into word-like groups and some groups are actual English words, but most are not.
* Some of the words in sequence make sense, although most are gibberish.

## 4) Training on more LSTM layers
Try to improve the quality of the generated text by creating a much larger network.
<br/>
Keeping the settings the same but add a stack of alternated LSTM and Dropout layers. Also increase the number of training epochs from 20 to 30 and decrease the batch size from 128 to 64 to give the network more opportunity to be updated and learn.
### Define the LSTM model
```python
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
### Fit the model
```python
model.fit(X, y, epochs=30, batch_size=64)
```
{% highlight text %}
Epoch 1/30
144245/144245 [==============================] - 1358s 9ms/step - loss: 2.7445
Epoch 2/30
144245/144245 [==============================] - 1361s 9ms/step - loss: 2.4729
Epoch 3/30
144245/144245 [==============================] - 1359s 9ms/step - loss: 2.2791
Epoch 4/30
144245/144245 [==============================] - 1284s 9ms/step - loss: 2.1366
Epoch 5/30
144245/144245 [==============================] - 1310s 9ms/step - loss: 2.0303
Epoch 6/30
144245/144245 [==============================] - 1290s 9ms/step - loss: 1.9477
Epoch 7/30
144245/144245 [==============================] - 1290s 9ms/step - loss: 1.8777
Epoch 8/30
144245/144245 [==============================] - 1335s 9ms/step - loss: 1.8214
Epoch 9/30
144245/144245 [==============================] - 1299s 9ms/step - loss: 1.7726
Epoch 10/30
144245/144245 [==============================] - 1291s 9ms/step - loss: 1.7257
Epoch 11/30
144245/144245 [==============================] - 1292s 9ms/step - loss: 1.6871
Epoch 12/30
144245/144245 [==============================] - 1291s 9ms/step - loss: 1.6503
Epoch 13/30
144245/144245 [==============================] - 1292s 9ms/step - loss: 1.6213
Epoch 14/30
144245/144245 [==============================] - 1291s 9ms/step - loss: 1.5894
Epoch 15/30
144245/144245 [==============================] - 1291s 9ms/step - loss: 1.5649
Epoch 16/30
144245/144245 [==============================] - 1326s 9ms/step - loss: 1.5395
Epoch 17/30
144245/144245 [==============================] - 1421s 10ms/step - loss: 1.5163
Epoch 18/30
144245/144245 [==============================] - 1301s 9ms/step - loss: 1.4939
Epoch 19/30
144245/144245 [==============================] - 1302s 9ms/step - loss: 1.4749
Epoch 20/30
144245/144245 [==============================] - 1313s 9ms/step - loss: 1.4575
Epoch 21/30
144245/144245 [==============================] - 1347s 9ms/step - loss: 1.4426
Epoch 22/30
144245/144245 [==============================] - 1296s 9ms/step - loss: 1.4262
Epoch 23/30
144245/144245 [==============================] - 1355s 9ms/step - loss: 1.4078
Epoch 24/30
144245/144245 [==============================] - 1394s 10ms/step - loss: 1.3964
Epoch 25/30
144245/144245 [==============================] - 1295s 9ms/step - loss: 1.3842
Epoch 26/30
144245/144245 [==============================] - 1292s 9ms/step - loss: 1.3743
Epoch 27/30
144245/144245 [==============================] - 1292s 9ms/step - loss: 1.3619
Epoch 28/30
144245/144245 [==============================] - 1292s 9ms/step - loss: 1.3513
Epoch 29/30
144245/144245 [==============================] - 1309s 9ms/step - loss: 1.3456
Epoch 30/30
144245/144245 [==============================] - 1295s 9ms/step - loss: 1.3376
{% endhighlight %} 

### Pick a random seed sequence
```python
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
```
{% highlight text %}
Seed:
"  favourite word 'moral,' and the arm that was linked
into hers began to tremble. alice looked up, an "
{% endhighlight %} 


### Generate characters
```python
for i in range(800):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
```
{% highlight text %}
d she was to tery toietier. 'the dormouse suireled of the batter- and the thought it had fome 
to be a little before she had not thing to her one of the court, but she had not ao an a large 
crom that iad a wasch out of the way it was a little before she had not thing to her on the way 
the way the way the way the way the white rabbit, which was suite a little shriel, and she white 
rabbit replied, 'it sas all the batter-'

'i don't know it as all the same thing as all ' said the mock turtle. 'but i'sh aegan to the dance.

                                                                             *    *    *    *    * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  * 
{% endhighlight %} 
Note some observations about the generate text:
* There are slightly lesser spelling mistakes.
* Some phrases gets repeated again and again like "the way" and "as all".
* "*" seems to be generated repeatedly.

## 5) Training on processed text
Try to improve the quality of the generated text by removing all punctuation and non alphabetical characters from the source text.
<br/>
Keeping the settings the same as the previous network but increase the number of training epochs from 30 to 40.
### Remove any punctuation
```python
import string
text = text.translate(str.maketrans('', '', string.punctuation))
```

### Remove any remaining non alphabetic characters
Some non alphabetic characters are not part of string.punctuation dictionary, we have to remove it separately.
```python
# \n indicating a new line, therefore we replace it with a space
text = text.replace('\n', ' ')
```
```python
symbols = ['»', '¿', 'ï']

for symbol in symbols:
    text = text.replace(symbol, "")
```

### Viewing the first 1000 characters
```python
text[:1000]
```
{% highlight text %}
'chapter i down the rabbithole  alice was beginning to get very tired of sitting by her 
sister on the bank and of having nothing to do once or twice she had peeped into the book 
her sister was reading but it had no pictures or conversations in it and what is the use 
of a book thought alice without pictures or conversations  so she was considering in her 
own mind as well as she could for the hot day made her feel very sleepy and stupid whether 
the pleasure of making a daisychain would be worth the trouble of getting up and picking 
the daisies when suddenly a white rabbit with pink eyes ran close by her  there was nothing 
so very remarkable in that nor did alice think it so very much out of the way to hear the 
rabbit say to itself oh dear oh dear i shall be late when she thought it over afterwards 
it occurred to her that she ought to have wondered at this but at the time it all seemed 
quite natural but when the rabbit actually took a watch out of its waistcoatpocket and 
looked at it and t'
{% endhighlight %} 
Now the text looks much cleaner.

### Redefine the mapping and reverse mapping
```python
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
```

### Display the unique vocabulary
```python
print(chars)
```
{% highlight text %}
[' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
{% endhighlight %} 
Now what's left is all alphabetical and a spacing.

### Summarize the loaded data
```python
n_chars = len(text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
```
{% highlight text %}
Total Characters:  136024
Total Vocab:  27
{% endhighlight %} 
Now the book have a total of 136,024 characters and when converted to lowercase there are 27 distinct characters in the vocabulary.

### Prepare the dataset of input to output pairs encoded as integers
```python
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
```
{% highlight text %}
Total Patterns:  135924
{% endhighlight %} 
The total number of patterns is 135,924.

### Reshape into the form [samples, time steps, features]
```python
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
```

### Normalization
```python
X = X / float(n_vocab)
```

### One hot encode
```python
y = np_utils.to_categorical(dataY)
```

### Define the LSTM model
```python
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

### Fit the model
```python
model.fit(X, y, epochs=40, batch_size=64)
```
{% highlight text %}
Epoch 1/40
135924/135924 [==============================] - 1305s 10ms/step - loss: 2.6210
Epoch 2/40
135924/135924 [==============================] - 1344s 10ms/step - loss: 2.1962
Epoch 3/40
135924/135924 [==============================] - 1237s 9ms/step - loss: 1.9830
Epoch 4/40
135924/135924 [==============================] - 1217s 9ms/step - loss: 1.8521
Epoch 5/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.7576
Epoch 6/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.6859
Epoch 7/40
135924/135924 [==============================] - 1218s 9ms/step - loss: 1.6249
Epoch 8/40
135924/135924 [==============================] - 1235s 9ms/step - loss: 1.5740
Epoch 9/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.5298
Epoch 10/40
135924/135924 [==============================] - 1211s 9ms/step - loss: 1.4919
Epoch 11/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.4580
Epoch 12/40
135924/135924 [==============================] - 1225s 9ms/step - loss: 1.4276
Epoch 13/40
135924/135924 [==============================] - 1230s 9ms/step - loss: 1.3983
Epoch 14/40
135924/135924 [==============================] - 1276s 9ms/step - loss: 1.3724
Epoch 15/40
135924/135924 [==============================] - 1234s 9ms/step - loss: 1.3454
Epoch 16/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.3281
Epoch 17/40
135924/135924 [==============================] - 1213s 9ms/step - loss: 1.3093
Epoch 18/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.2907
Epoch 19/40
135924/135924 [==============================] - 1213s 9ms/step - loss: 1.2719
Epoch 20/40
135924/135924 [==============================] - 1214s 9ms/step - loss: 1.2589
Epoch 21/40
135924/135924 [==============================] - 1213s 9ms/step - loss: 1.2437
Epoch 22/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.2360
Epoch 23/40
135924/135924 [==============================] - 1231s 9ms/step - loss: 1.2185
Epoch 24/40
135924/135924 [==============================] - 1213s 9ms/step - loss: 1.2089
Epoch 25/40
135924/135924 [==============================] - 1213s 9ms/step - loss: 1.2011
Epoch 26/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.1898
Epoch 27/40
135924/135924 [==============================] - 1211s 9ms/step - loss: 1.1810
Epoch 28/40
135924/135924 [==============================] - 1221s 9ms/step - loss: 1.1804
Epoch 29/40
135924/135924 [==============================] - 1219s 9ms/step - loss: 1.1700
Epoch 30/40
135924/135924 [==============================] - 1224s 9ms/step - loss: 1.1622
Epoch 31/40
135924/135924 [==============================] - 1214s 9ms/step - loss: 1.1577
Epoch 32/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.1536
Epoch 33/40
135924/135924 [==============================] - 1282s 9ms/step - loss: 1.1469
Epoch 34/40
135924/135924 [==============================] - 1249s 9ms/step - loss: 1.1396
Epoch 35/40
135924/135924 [==============================] - 1212s 9ms/step - loss: 1.1402
Epoch 36/40
135924/135924 [==============================] - 1220s 9ms/step - loss: 1.1325
Epoch 37/40
135924/135924 [==============================] - 1210s 9ms/step - loss: 1.1277
Epoch 38/40
135924/135924 [==============================] - 1211s 9ms/step - loss: 1.1250
Epoch 39/40
135924/135924 [==============================] - 1211s 9ms/step - loss: 1.1231
Epoch 40/40
135924/135924 [==============================] - 1226s 9ms/step - loss: 1.1254
{% endhighlight %} 

### Pick a random seed sequence
```python
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
```
{% highlight text %}
Seed:
"  learn  well there was mystery the mock turtle replied counting off the subjects on his flappers mys "
{% endhighlight %} 


### Generate characters
```python
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
```
{% highlight text %}
t beginning to the beginning to the babk to the thing and she was so she went on and then and 
then alice thought alice they were all that said the caterpillar  well i she said the gryphon  
the dormouse was spoetely and then they lay sp they had a conversation and there was a long 
silence and then the mock turtle said to the jury  she mock turtle senlied  there was a long 
silence and then all the thing and she was so saying to herself in a moment that it was a 
little befin and then all the court and the mock turtle said the queen  i dont know that she 
was a very gind her something was and she said to herself i shall be a call i can rail iis 
head out of the words a little bett that she was a very gind her something was and she said 
to herself in a moment that it was a little befin and then all the court and the mock turtle 
said the queen  i dont know that she was a very gind her something was and she said to herself 
i shall be a call i can rail iis head out of the words a little bett tha
{% endhighlight %} 
Note some observations about the generate text:
* There are much lesser spelling mistakes and the text looks more realistic.
* There is less repetition of phrases.
* Overall the text is still making not much sense.

## 6) Conclusion
With a sufficiently trained model, we could generate realistic text that are more language-like and legible, which at one glance could be difficult to be differentiated from actual written text by humans.
<br/>

To further improve the model, we could try tuning the model, such as the number of epochs, memory units, batch size or even the dropout percentage to see if we can develop a better model. We could also explore developing a word-based language model rather than a character-based model.

