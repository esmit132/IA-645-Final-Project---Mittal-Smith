#!/usr/bin/env python
# coding: utf-8

# In[1]:


# key sources: https://harrisonjansma.com/apple

import pandas as pd
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# read in data (apple)
apple = pd.read_csv("C:/Users/Ethan/Desktop/IA 645 Project/Apple-Twitter-Sentiment-DFE.csv")
apple


# In[2]:


# read in data (airline)
air = pd.read_csv("C:/Users/Ethan/Desktop/IA 645 Project/AirlineTweets.csv")
#air


# In[3]:


# create list of binary labels by defining a function: 0=negative (1s) and 1=positive (3s and 5s)
# apple
def binarizer(s):

    if s=='1':
        return 0
    else:
        return 1
    
binary_labels = apple['sentiment'].apply(binarizer)
# returns pandas.core.series.Series object 
binary_labels
# convert to data frame object
binary_labels = binary_labels.to_frame()

# insert into the dataframe at the beginning
apple.insert(0,'binary_value',binary_labels)
apple

# airline
def son_of_binarizer(t):
    
    if t=='negative':
        return 0
    else:
        return 1
    
binary_labels2 = air['airline_sentiment'].apply(son_of_binarizer)
binary_labels2
binary_labels2 = binary_labels2.to_frame()


# insert into the dataframe at the beginning
air.insert(0,'binary_value',binary_labels2)
air


# In[4]:


# add column for "postive, negative, and neutral" labels in 5,3,1 code (airline)

def numerizer(f):
    if f=='positive':
        return 5
    if f=='neutral':
        return 3
    else:
        return 1
# use function to get list of numeric labels    
numeric_labels = air['airline_sentiment'].apply(numerizer)
numeric_labels
numeric_labels = numeric_labels.to_frame()

# insert labels to dataframe at beginning
air.insert(0,'sentiment_score',numeric_labels)


# In[5]:


# binarize airline sentiment labels so that 'positive' and 'neutral' are 'positive' and 'negative' is 'negative' (airline)

def binary_labeler(q):
    if q=='negative':
        return "negative"
    else:
        return "positive"
# get list of binary text labels
binary_text_labels = air['airline_sentiment'].apply(binary_labeler)
binary_text_labels
binary_text_labels = binary_text_labels.to_frame()

# insert labels to dataframe at beginning
air.insert(0,'binary_labels',binary_text_labels)
air


# In[6]:


# add column for 5,3,1 code in "positive,negative, and neutral" labels (apple)

def labeler(z):
    if z=='1':
        return "negative"
    if z=='3':
        return "neutral"
    if z=='5':
        return "positive"
text_labels = apple['sentiment'].apply(labeler)
text_labels
text_labels = text_labels.to_frame()

# insert labels to dataframe at beginning
apple.insert(0,'apple_sentiment',text_labels)


# In[7]:


# add column for binary text labels (apple)

def labeler(z):
    if z=='1':
        return "negative"
    else:
        return "positive"
text_labels2 = apple['sentiment'].apply(labeler)
text_labels2
text_labels2 = text_labels2.to_frame()

# insert labels to dataframe at beginning
apple.insert(0,'binary_labels',text_labels2)


# In[8]:


# rename columns to prepare for dropping (airline)
air = air. rename(columns={'tweet_id':'A'})
air = air. rename(columns={'negativereason':'B'})
air = air. rename(columns={'negativereason_confidence':'C'})
air = air. rename(columns={'airline':'D'})
air = air. rename(columns={'airline_sentiment_gold':'E'})
air = air. rename(columns={'name':'F'})
air = air. rename(columns={'negativereason_gold':'G'})
air = air. rename(columns={'retweet_count':'H'})
air = air. rename(columns={'tweet_coord':'I'})
air = air. rename(columns={'tweet_created':'J'})
air = air. rename(columns={'tweet_location':'K'})
air = air. rename(columns={'user_timezone':'L'})
air


# In[9]:


# drop unused columns (airline)
air=air.drop('A',axis=1)
air=air.drop('B',axis=1)
air=air.drop('C',axis=1)
air=air.drop('D',axis=1)
air=air.drop('E',axis=1)
air=air.drop('F',axis=1)
air=air.drop('G',axis=1)
air=air.drop('H',axis=1)
air=air.drop('I',axis=1)
air=air.drop('J',axis=1)
air=air.drop('K',axis=1)
air=air.drop('L',axis=1)
air


# In[10]:


# rename columns to prepare to drop (apple)
apple = apple. rename(columns={'_unit_id':'a'})
apple = apple. rename(columns={'_golden':'b'})
apple = apple. rename(columns={'_unit_state':'c'})
apple = apple. rename(columns={'_trusted_judgments':'d'})
apple = apple. rename(columns={'_last_judgment_at':'e'})
apple = apple. rename(columns={'sentiment_gold':'f'})
apple


# In[11]:


#Drop all columns except sentiment scores, sentiment labels, sentiment confidence, and tweet text (axis=1 means drop columns, not index) # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html?highlight=drop#pandas.DataFrame.drop
apple=apple.drop('date',axis=1)
apple=apple.drop('query',axis=1)
apple=apple.drop('id',axis=1)
apple=apple.drop('b',axis=1)
apple=apple.drop('a',axis=1)
apple=apple.drop('c',axis=1)
apple=apple.drop('d',axis=1)
apple=apple.drop('e',axis=1)
apple=apple.drop('f',axis=1)
apple


# In[12]:


#Delete all rows where 'not_relevant' is listed under sentiment (apple)
apple = apple[apple['sentiment']!='not_relevant']
apple


# In[13]:


# remove all tweets that are "not_relevant" (airline)
air = air[air['airline_sentiment']!='not_relevant']
air


# In[14]:


# print a graph that shows the distribution of negative, neutral, and positive tweets where 1=negative, 3=neutal, and 5=positive
# source: https://mode.com/python-tutorial/counting-and-plotting-in-python/

# apple
sentiment_bar_graph = apple['apple_sentiment'].value_counts().plot(kind='bar',title='Sentiment vs. # of Tweets')


# In[15]:


# airline
sentiment_bar_graph2 = air['airline_sentiment'].value_counts().plot(kind='bar',title='Sentiment vs. # of Tweets')


# In[16]:


# graph of positive vs. negative tweets in binary split
sentiment_bar_graph3 = apple['binary_labels'].value_counts().plot(kind='bar',title='Apple - Binary Sentiment vs. # of Tweets')

# get exact counts
print(apple.sentiment.value_counts())
print(apple.binary_value.value_counts())


# In[17]:


sentiment_bar_graph4 = air['binary_labels'].value_counts().plot(kind='bar',title='Airline - Binary Sentiment vs. # of Tweets')
print(air.airline_sentiment.value_counts())
print(air.binary_value.value_counts())


# In[18]:


# standardize dataset by removing non-text symbols and converting to LC # https://harrisonjansma.com/apple#Symbols-to-be-removed.

def remove_symbols(apple,textX):
    apple[textX] = apple[textX].str.replace(r"http\S+","")
    apple[textX] = apple[textX].str.replace(r"http","")
    apple[textX] = apple[textX].str.replace(r"@\S+","")
    apple[textX] = apple[textX].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]"," ")
    apple[textX] = apple[textX].str.replace(r"@","at")
    apple[textX] = apple[textX].str.replace(r"'","")
    apple[textX] = apple[textX].str.lower()
    return apple

apple = remove_symbols(apple,'text')
apple


# In[19]:


def remove_symbols2(air,textY):
    air[textY] = air[textY].str.replace(r"http\S+","")
    air[textY] = air[textY].str.replace(r"http","")
    air[textY] = air[textY].str.replace(r"@\S+","")
    air[textY] = air[textY].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]"," ")
    air[textY] = air[textY].str.replace(r"@","at")
    air[textY] = air[textY].str.replace(r"'","")
    air[textY] = air[textY].str.lower()
    return air

air = remove_symbols2(air,'text')
air


# In[20]:


#conda install -c conda-forge nltk


# In[21]:


# tokenize data to convert each tweet into a list of individual tokens and delete any empty tokens 
# source: https://harrisonjansma.com/apple#Symbols-to-be-removed.
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+',discard_empty=1)
apple['tokens'] = apple['text'].apply(tokenizer.tokenize)

tokenz = apple['tokens']
tokenz.to_frame()


# In[22]:


tokenizer2 = RegexpTokenizer(r'\w+',discard_empty=1)
air['tokens2'] = air['text'].apply(tokenizer2.tokenize)

tokenz2 = air['tokens2']
tokenz2.to_frame()
air


# In[23]:


import matplotlib.pyplot as plt
sentence_lengths = [len(tokens) for tokens in apple['tokens']]
vocab = sorted(list(set([word for tokens in apple['tokens'] for word in tokens])))

plt.figure(figsize = (10,10))
plt.xlabel('Sentence Length (in words)')
plt.ylabel('Number of Tweets')
plt.title('Sentence Lengths')
plt.hist(sentence_lengths)
plt.show()


# In[24]:


sentence_lengths = [len(tokens2) for tokens2 in air['tokens2']]
vocab = sorted(list(set([word for tokens2 in air['tokens2'] for word in tokens2])))

plt.figure(figsize = (10,10))
plt.xlabel('Sentence Length (in words)')
plt.ylabel('Number of Tweets')
plt.title('Sentence Lengths')
plt.hist(sentence_lengths)
plt.show()


# In[25]:


### Get the most common words in negative apple tweets

# isolate all apple tweet text
apple_text = apple['text']
# get only negative apple tweets
bad_apple_df = apple[apple['binary_value'] == 0]
bad_apple = bad_apple_df['text']
bad_apple

from sklearn.feature_extraction.text import CountVectorizer

# get word,frequency pairs in negative apple tweets as list
texts = bad_apple
vec = CountVectorizer().fit(texts)
bag_of_words = vec.transform(texts)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
sorted(words_freq, key = lambda x: x[1], reverse=True)

# convert list to 
type(words_freq)

words_freq

# convert list to dataframe
words_freq_df = pd.DataFrame(data=words_freq).T   
words_freq_df.columns = words_freq_df.iloc[0]    
words_freq_df = words_freq_df.iloc[1:]  

# transpose dataframe
words_freq_df = words_freq_df.transpose()

# convert to columns and add labels 
words_freq_df.columns = ['Frequency']
words_freq_df['Term'] = words_freq_df.index

words_freq_df.reset_index(level=0,inplace=True)
words_freq_df = words_freq_df.drop(words_freq_df.columns[0],axis=1)

words_freq_df = words_freq_df.sort_values('Frequency',ascending=False)
words_freq_df

# get top 10 most common words in negative apple tweets without taking out stop words or repeated words
top10_bad_apple = words_freq_df[0:10]
top10_bad_apple


# In[26]:


import pandas as pd
import matplotlib.pyplot as plot
# Draw a vertical bar chart of most common words

top10_bad_apple.plot.bar(x="Term", y="Frequency", rot=70, title="Most Common Words in Negative Apple Tweets");

plot.show(block=True);


# In[27]:


### Get TF-IDF scores for 10 Keywords
# tfidf score shows the 'importance' of words, not just how common they are ('and' might be the most common word but TF-IDF takes into account the uniqueness of the word)

# convert series of only negative apple tweets to list
bad_apple_list = bad_apple.tolist()
# convert list to single string
bad_apple_str = ''.join(bad_apple_list)


# In[28]:


# get stop words (common words that should be filtered out like 'a')
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# create a vocabulary of words from only negative apple tweets that ignores stop words and words that appear in 85% of the tweets
countVect = CountVectorizer(max_df=0.85,stop_words=stopWords)
word_out = countVect.fit_transform(apple_text)


# In[29]:


# define functions to sort keywords by their ID-IDF scores in descending order
#source: https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XycvcuuSlPY
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the words and tf-idf score of top 10 items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of word and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# In[30]:


# generate keywords and tfifd. source: https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.Xyb4TuuSlPY

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transf=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transf.fit(word_out)

# match index to "feature_names"
feature_names=countVect.get_feature_names()

#generate tf-idf for negative apple tweets
tf_idf_vector=tfidf_transf.transform(countVect.transform([bad_apple_str]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract top 10 as dict
bad_apple_keywords=extract_topn_from_vector(feature_names,sorted_items,10)
# convert dict to dataframe
bak_items = bad_apple_keywords.items()
bak_list = list(bak_items)
bak_df = pd.DataFrame(bak_list)

# rename columns
bak_df.columns = ['Term','TF-IDF Score']
bak_df


# In[46]:


### repeat to get keywords and ID-IDF scores for positive apple, negative airline, and positive airline

## positive apple

# create a vocabulary of words from only positive apple tweets that ignores stop words and words that appear in 85% of the tweets
cv2 = CountVectorizer(max_df=0.85,stop_words=stopWords)
wo2 = cv2.fit_transform(apple_text)
tfidf_transf2 = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transf.fit(word_out)

# match index to "feature_names"
feature_names2=cv2.get_feature_names()

# get positive apple tweets as string
good_apple_df = apple[apple['binary_value'] == 1]
good_apple = good_apple_df['text']
good_apple_list = good_apple.tolist()
good_apple_str = ''.join(good_apple_list)

#generate tf-idf for positive apple tweets
tf_idf_vector2=tfidf_transf.transform(cv2.transform([good_apple_str]))

#sort the tf-idf vectors by descending order of scores
sorted_items2=sort_coo(tf_idf_vector2.tocoo())

#extract top 10 as dict
good_apple_keywords=extract_topn_from_vector(feature_names2,sorted_items2,10)
# convert dict to dataframe
gak_items = good_apple_keywords.items()
gak_list = list(gak_items)
gak_df = pd.DataFrame(gak_list)

# rename columns
gak_df.columns = ['Term','TF-IDF Score']
gak_df


# In[48]:


## negative airline

# get all airline tweet text 
air_text = air['text']

# create a vocabulary of words from only positive apple tweets that ignores stop words and words that appear in 85% of the tweets
cv3 = CountVectorizer(max_df=0.85,stop_words=stopWords)
wo3 = cv3.fit_transform(air_text)
tfidf_transf3 = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transf.fit(wo3)

# match index to "feature_names"
feature_names3=cv3.get_feature_names()

# get negative airline tweets as string
bad_air_df = air[air['binary_value']==0]
bad_air = bad_air_df['text']
bad_air_list = bad_air.tolist()
bad_air_str = ''.join(bad_air_list)

#generate tf-idf for positive apple tweets
tf_idf_vector3=tfidf_transf.transform(cv3.transform([bad_air_str]))

#sort the tf-idf vectors by descending order of scores
sorted_items3=sort_coo(tf_idf_vector3.tocoo())

#extract top 10 as dict
bad_air_keywords=extract_topn_from_vector(feature_names3,sorted_items3,10)
# convert dict to dataframe
baik_items = bad_air_keywords.items()
baik_list = list(baik_items)
baik_df = pd.DataFrame(baik_list)

# rename columns
baik_df.columns = ['Term','TF-IDF Score']
baik_df


# In[53]:


## positive airline

# create a vocabulary of words from only positive airline tweets that ignores stop words and words that appear in 85% of the tweets
cv4 = CountVectorizer(max_df=0.85,stop_words=stopWords)
wo4 = cv4.fit_transform(air_text)
tfidf_transf4 = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transf.fit(wo4)

# match index to "feature_names"
feature_names4=cv4.get_feature_names()

# get negative airline tweets as string
good_air_df = air[air['binary_value']==1]
good_air = good_air_df['text']
good_air_list = good_air.tolist()
good_air_str = ''.join(good_air_list)

#generate tf-idf for positive apple tweets
tf_idf_vector4=tfidf_transf.transform(cv4.transform([good_air_str]))

#sort the tf-idf vectors by descending order of scores
sorted_items4=sort_coo(tf_idf_vector4.tocoo())

#extract top 10 as dict
good_air_keywords=extract_topn_from_vector(feature_names4,sorted_items4,10)
# convert dict to dataframe
gaik_items = good_air_keywords.items()
gaik_list = list(gaik_items)
gaik_df = pd.DataFrame(gaik_list)

# rename columns
gaik_df.columns = ['Term','TF-IDF Score']
gaik_df

