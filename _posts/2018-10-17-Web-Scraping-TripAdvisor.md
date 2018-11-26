---
title: Web Scraping TripAdvisor and Text Mining
date: 2018-10-17
tags: 
  - Web Scraping
  - NLP
header:
  image: ""
excerpt: "Web Scraping, NLP"
mathjax: "true"
---
These days lots of unstructured data is available on the internet, the dataset used will be extracted from TripAdvisor. 
TripAdvisor is a widely known website that shows hotel and restaurant reviews, accommodation bookings and other travel-related content.
<br/>
I performed a web scraping on TripAdvisor for the hotel, Marina Bay Sands, using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) and an example from [furas](https://github.com/furas). The data would then be contained in a csv.




<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/page1.png" alt="">

The items to be retrieved are:
* Content of the reviews
* Ratings of the reviews
* Reviewed date

<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/page2.png" alt="">

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
mbs = pd.read_csv('mbs.csv', sep=',')
mbs.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>review_body</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>I stayed for one night here.I decide to upgrad...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Yes it’s huge, yes it’s expensive but honestly...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Iconic hotel very good public areas staff are ...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>We stayed 1 night in this hotel because I want...</td>
      <td>November 7, 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Very helpful staff during checkin. Room was cl...</td>
      <td>November 7, 2018</td>
    </tr>
  </tbody>
</table>

### View the shape of the dataframe
```python
mbs.shape
```
{% highlight text %}
(16682, 4)
{% endhighlight %} 

### View a review content
```python
mbs["review_body"][50]
```
{% highlight text %}
'We loved staying here! We splurged for our 5-year anniversary since we are usually budget 
travellers. You must wake up early to take good pictures. We woke up around 6:30 am and 
took great pictures with good lighting and without many people! We had room service which 
was expensive but quite good. We just split 1 breakfast and it was enough for 2 people. 
You come here for the infinity pool and to take good pictures, so enjoy!'
{% endhighlight %} 

### Change the datetime format
```python
mbs['review_date'] = pd.to_datetime(mbs['review_date'])
mbs.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>review_body</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>I stayed for one night here.I decide to upgrad...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Yes it’s huge, yes it’s expensive but honestly...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Iconic hotel very good public areas staff are ...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>We stayed 1 night in this hotel because I want...</td>
      <td>2018-11-07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Very helpful staff during checkin. Room was cl...</td>
      <td>2018-11-07</td>
    </tr>
  </tbody>
</table>

## 2) Exploratory data analysis
### Count of the total ratings
```python
sns.countplot(y=mbs['rating'], color='mediumseagreen', order=[5,4,3,2,1])
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot.png" alt="">
<br/>
Indeed the distribution of the total ratings is similar to the one on the TripAdvisor website.

### Count of the no. of reviews in each year
```python
sns.countplot(x = 'year', data=mbs, palette="GnBu_d")
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot2.png" alt="">
<br/>
Noting that the difference seen in 2010 is only because Marina Bay Sands had a soft opening on 27 April 2010 and had it's grand opening on 17 February 2011.

### Count of the unique ratings against each year
Create a new dataframe column by segregating review date by year
```python
mbs['year'] = mbs['review_date'].dt.year
```
Countplot of the individual ratings per year
```python
plt.figure(figsize=(12,8)) 
ax= sns.countplot(x='year' ,hue='rating',data=mbs, palette="Set3")
ax.set(xlabel='Year', ylabel='Count')
ax.figure.suptitle("Ratings by Year", fontsize = 20)
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/countplot3.png" alt="">
* Negative ratings peaks in it's initial launch in year 2010.
* The counts of negative ratings remains quite constant throughout, but reducing in proportion when compared against individual yearly total rating counts.
* Positive ratings gradually starts to increase from year 2011 amidst it's grand opening and peaks in the year 2016.
* Positive ratings starts to experience a decline from year 2017 following it's peak in year 2016.

## 3) Bag-of-words
Create a set of features indicating the number of times a text contains a particular word, using a bag of words model. Bag of words models output a feature for every unique word in text data, with each feature containing a count of occurrences in observations. The output would results in a matrix that can contain thousands of features and it would likely be a sparse matrix as most words do not occur in most observations.
### Define a function for processing the text
```python
import string
from nltk.corpus import stopwords

def text_process(text):
    # Remove any punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any stopwords
    # NLTK's stopwords assumes words are all lowercased
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    #Join the characters again to form the string
    return " ".join(text)
```

### Apply the function on the text
```python
reviews = mbs['review_body'].apply(text_process)
reviews.head()
```
{% highlight text %}
0    stayed one night hereI decide upgrade Club55 a...
1    Yes it’s huge yes it’s expensive honestly woul...
2    Iconic hotel good public areas staff amazing l...
3    stayed 1 night hotel wanted swim rooftop pool ...
4    helpful staff checkin Room clean bright Given ...
Name: review_body, dtype: object
{% endhighlight %} 

### Creating the bag of words
```python
from sklearn.feature_extraction.text import CountVectorizer
# Create a bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(reviews)
```

### Plot the 30 most common words
```python
import collections

word_freq = dict(zip(count.get_feature_names(), np.asarray(bag_of_words.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(30), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
#sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
sns.barplot(x="freq", y="word", data=word_counter_df, palette="Blues_d", ax=ax, orient="h")
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Most Common 30 Words')
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/barplot.png" alt="">
The vocabulary contains singular and plural versions of some words, such as "room" and "rooms". The vocabulary also includes words like "stay" and "stayed" which are different verbs forms and a noun relating to the verb "to stay". Having such issues is disadvantageous for building a model that generalizes well. 
<br/>
It could be overcome by methods of stemming, which is to drop common suffixes and identifying all the words that have the same word stem. Or by lemmatization, which instead uses a dictionary of known word forms and the role of the word in the sentence is taken into account.

## 4) N-grams
One of the main disadvantages of using a bag-of-words representation is that word order is completely discarded. There is a way of capturing context when using bag-of-words representation, by using the method of n-grams. It will not only consider the counts of single tokens, but also the counts of pairs or triplets of tokens that appear next to each other.
### Create n-grams functions
```python
from nltk.util import ngrams
## Helper Functions
def get_ngrams(text, n):
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]
```
```python
from collections import Counter
def gramfreq(text,n,num):
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)
    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name
    return df.sort_values(["frequency"],ascending=[0])[:num]
```
### Display the results of n-grams on a table
```python
def gram_table(text, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramfreq(preprocessing(text),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Frequency"]
        out = pd.concat([out, table], axis=1)
    return out
```
### Text preprocessing with stemming
```python
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
stop_words = set(stopwords.words('english'))
def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ')
    words = word_tokenize(txt)
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    return words
```
### Execute the n-grams function
Generating 4-grams(unigrams, bigrams, trigrams and quadgrams)
```python
gram_table(reviews, gram=[1,2,3,4], length=15)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1-Gram</th>
      <th>Frequency</th>
      <th>2-Gram</th>
      <th>Frequency</th>
      <th>3-Gram</th>
      <th>Frequency</th>
      <th>4-Gram</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hotel</td>
      <td>31203</td>
      <td>marina bay</td>
      <td>5649</td>
      <td>marina bay sand</td>
      <td>4216</td>
      <td>stay marina bay sand</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>1</th>
      <td>room</td>
      <td>30514</td>
      <td>infin pool</td>
      <td>5315</td>
      <td>stay marina bay</td>
      <td>1167</td>
      <td>marina bay sand hotel</td>
      <td>474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pool</td>
      <td>21737</td>
      <td>bay sand</td>
      <td>4327</td>
      <td>5 star hotel</td>
      <td>664</td>
      <td>infin pool 57th floor</td>
      <td>211</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stay</td>
      <td>18657</td>
      <td>garden bay</td>
      <td>2066</td>
      <td>citi view room</td>
      <td>508</td>
      <td>night marina bay sand</td>
      <td>203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>view</td>
      <td>16684</td>
      <td>citi view</td>
      <td>1959</td>
      <td>stay one night</td>
      <td>491</td>
      <td>marina bay sand one</td>
      <td>118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>servic</td>
      <td>9289</td>
      <td>club room</td>
      <td>1801</td>
      <td>bay sand hotel</td>
      <td>485</td>
      <td>visit marina bay sand</td>
      <td>91</td>
    </tr>
    <tr>
      <th>6</th>
      <td>check</td>
      <td>9209</td>
      <td>swim pool</td>
      <td>1753</td>
      <td>view garden bay</td>
      <td>434</td>
      <td>singapor marina bay sand</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>singapor</td>
      <td>9081</td>
      <td>stay hotel</td>
      <td>1651</td>
      <td>pool 57th floor</td>
      <td>414</td>
      <td>club room citi view</td>
      <td>85</td>
    </tr>
    <tr>
      <th>8</th>
      <td>night</td>
      <td>9021</td>
      <td>view room</td>
      <td>1462</td>
      <td>room citi view</td>
      <td>356</td>
      <td>marina bay sand singapor</td>
      <td>84</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bay</td>
      <td>9000</td>
      <td>one night</td>
      <td>1421</td>
      <td>ku de ta</td>
      <td>342</td>
      <td>experi marina bay sand</td>
      <td>76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>one</td>
      <td>8930</td>
      <td>5 star</td>
      <td>1259</td>
      <td>stay 2 night</td>
      <td>283</td>
      <td>marina bay sand stay</td>
      <td>73</td>
    </tr>
    <tr>
      <th>11</th>
      <td>great</td>
      <td>8406</td>
      <td>stay marina</td>
      <td>1208</td>
      <td>infin pool amaz</td>
      <td>273</td>
      <td>hotel marina bay sand</td>
      <td>69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>time</td>
      <td>8078</td>
      <td>57th floor</td>
      <td>1188</td>
      <td>roof top pool</td>
      <td>259</td>
      <td>roof top infin pool</td>
      <td>63</td>
    </tr>
    <tr>
      <th>13</th>
      <td>staff</td>
      <td>7921</td>
      <td>shop mall</td>
      <td>1131</td>
      <td>rooftop infin pool</td>
      <td>254</td>
      <td>room face garden bay</td>
      <td>63</td>
    </tr>
    <tr>
      <th>14</th>
      <td>get</td>
      <td>7783</td>
      <td>great view</td>
      <td>1125</td>
      <td>infin pool 57th</td>
      <td>239</td>
      <td>swim pool 57th floor</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
Through the n-grams method we could see that the content of most reviews mentioned:
* "infinity pool" and through the 3-grams and 4-grams sequences, could conclude that the pool is on the 57th floor rooftop.
* "city view" and through the 3-grams and 4-grams sequences, could draw the conclusion that it was the club room with city view.
* "shop mall" through the 2-grams sequences, which is the shopping mall, The Shoppes at Marina Bay Sands.
* "ku de ta" which is the name of the formerly known bar in Marina Bay Sands.
* "garden bay" which is a nature park, Gardens by the Bay.

## 5) Topic Modeling with Latent Dirichlet Allocation
Topic modeling is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents and is frequently used for discovery of hidden semantic structures in a text body. Latent Dirichlet Allocation is an example of topic modeling and it allows sets of observations to be explained by unobserved groups, by finding groups of words (the topics) that appear together frequently.
### Create topic modeling
```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=20, learning_method="batch", max_iter=25, random_state=0)
# Build the model and transform the data in one step
# Computing transform takes some time, and we can save time by doing both at once.
document_topics = lda.fit_transform(bag_of_words)
```
### View the shape of the components in LDA
The shape of the components is in the form of (n_topics, n_words)
```python
lda.components_.shape
```
{% highlight text %}
(20, 51339)
{% endhighlight %} 

### Create function to view the most important words for each topics
```python
# for each topic (a row in the components_), sort the features (ascending).
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer:
feature_names = np.array(count.get_feature_names())
```
This is a function adapted from the [mglearn](https://github.com/amueller/mglearn) library, for the book "Introduction to Machine Learning with Python".
```python
def print_topics(topics, feature_names, sorting, topics_per_chunk, n_words):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")
```
### Generate out the 20 topics
```python
print_topics(topics=range(20), feature_names=feature_names, sorting=sorting, topics_per_chunk=6, n_words=10)
```
{% highlight text %}
topic 0       topic 1       topic 2       topic 3       topic 4       topic 5       
--------      --------      --------      --------      --------      --------      
room          balcony       kids          spore         hotel         marina        
pool          quite         even          hv            pool          bay           
view          doorstep      nice          skylark       singapore     sands         
amazing       deluxe        much          simply        bay           singapore     
great         inner         food          60th          view          stay          
night         pricy         chinese       creation      great         time          
check         chill         race          dax           marina        staff         
staff         room          place         minus         amazing       like          
stay          bayfront      family        prize         infinity      hotel         
us            cctv          come          rises         city          room          


topic 6       topic 7       topic 8       topic 9       topic 10      topic 11      
--------      --------      --------      --------      --------      --------      
casino        club          check         room          go            service       
mall          room          universal     hotel         dollars       dress         
diverse       suite         studios       would         even          code          
roads         floor         minute        mbs           floor         guest         
hotel         view          free          size          shangri       industry      
october       lounge        pleasant      service       sing          management    
frontdesk     city          cavalli       bed           smokers       looks         
notice        staff         roberto       stayed        passport      staffs        
luxury        breakfast     things        large         la            cannot        
ie            pool          different     well          charge        ppl           


topic 12      topic 13      topic 14      topic 15      topic 16      topic 17      
--------      --------      --------      --------      --------      --------      
birthday      der           room          hotel         baggages      room          
mbs           ist           view          pool          coupon        hotel         
stay          die           pool          stay          son           check         
cake          sehr          bathroom      like          blazer        us            
us            und           bay           people        capsules      service       
anniversary   das           nice          rooms         old           staff         
thank         ein           hotel         service       christmas     one           
service       mbs           also          room          uncle         would         
special       laundry       infinity      view          arround       told          
team          pad           bath          one           vip           get           


topic 18      topic 19      
--------      --------      
room          service       
hotel         offered       
also          reward        
pool          bond          
food          teh           
get           theatre       
area          venue         
floor         it            
good          399           
tower         jacuzzi           
{% endhighlight %} 
* Quite a few topics seems to capture the hotel's facilities and services.
* Topic 12 seems to be on special occasion.
* Topic 13 seems to capture some very peculiar words.

### Plot topics with weightage
Another way to inspect the topics by seeing how much weight each topic gets overall, by summing the document_topics over all reviews. Here, each topic is named by the two most common words.
```python
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
topic_names = ["{:>2} ".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])]

# two column bar chart:
for col in [0, 1]:
    start = col * 10
    end = (col + 1) * 10
    ax[col].barh(np.arange(10), np.sum(document_topics, axis=0)[start:end])
    ax[col].set_yticks(np.arange(10))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 5000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/WebScrapingTripAdvisor/chart.png" alt="">




```python

```
{% highlight text %}

{% endhighlight %} 
