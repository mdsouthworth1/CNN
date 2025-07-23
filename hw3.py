#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors, word2vec
import gensim
import gensim.downloader
from gensim.test.utils import common_texts, get_tmpfile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cosine as cosDist
import numpy as np
import pandas as pd
import json
import re
import os


# In[2]:


def setModel(Model):

    global w2vModel

    w2vModel = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)

    #path = get_tmpfile('word2vec.model')


# In[ ]:


#model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")


# In[ ]:


def findPlagiarism(sentences, target):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    target = re.sub(r'[^a-z0-9\s]', '', target.lower())
    target = target.split()
    print(target)
    target_words = []
    target_words = [lemmatizer.lemmatize(word) for word in target if word not in stop_words]
    #words = set(words)

    target_vectors = [w2vModel[word] for word in target_words if word in w2vModel]
    target_vector = np.mean(target_vectors, axis = 0)

    #print(target_vector)

    words = []
    cosine_distance = []
    for sentence in sentences:
        #print(sentence)
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence.lower())
        sentence = sentence.split()
        words = [lemmatizer.lemmatize(word) for word in sentence if word not in stop_words]
        #print(words)
        vectors = [w2vModel[word] for word in words if word in w2vModel]
        avg_vector = np.mean(vectors, axis = 0)
        #print(avg_vector)
        cosine = cosDist(target_vector, avg_vector)
        #print(cosine)
        cosine_distance.append(cosine)

    target_index = np.argmin(cosine_distance)

    #print(target_index)
    return target_index


# ## Subreddit Classification

# In[29]:


def classifySubreddit_train(file):

    global lr_w2v
    global Scaler
    global le
    lr = LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 2000)

    #file = 'redditComments_train.jsonlist'
    with open(file,  encoding='utf-8') as json_file:
        training_set = [json.loads(line) for line in json_file]

    train_class = [line['subreddit'] for line in training_set]
    train_comments = [line['body'] for line in training_set]

    #pd.Series(subreddit_class).value_counts()

    x_train = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    url_pattern = r'\((https?://[^\s]+)\)'
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    #replacements = lambda text: ''.join(replacement_rules.get(char, char) for char in text)
    for comment in train_comments:
        for char, replacement in replacement_rules.items():
            comment = comment.replace(char, replacement)
        #print('\nnew comment line:', comment)
        #for word in words:
        #if re.match(r'http[s]?://', comment):
        #Extract urls and add to token list. Text has useful urls for training
        combined_tokens = []
        urls = re.findall(url_pattern, comment)
        for url in urls:
            comment = re.sub(r'https?:\/\/(www\.)?', ' ', comment)
            sub_domain = url.split('/')
            domain_name = sub_domain[0]
            path_words = sub_domain[1:] if len(sub_domain) > 1 else []
            #Split domain and rest of url. Domain will likely be class specific
            split_domain = domain_name.split('.')
            path_tokens = []
            for i in path_words:
                path_tokens.extend(re.split(r'[\/\-_+\)\(\.]', i))            
            split_url = split_domain + [token for token in path_tokens if token.strip()]
            #Take care of edge cases
            for j in split_url:
                if ' ' in j:
                    combined_tokens.extend(j.lower().split())
                else:
                    combined_tokens.append(j.lower())
            for word in combined_tokens:
                word = re.sub(r'\W', '', word.lower())
        #Remove processed urls or else duplicates
        comment = re.sub(url_pattern, '', comment)
        words = re.findall(r'\b\w+(?:\'\w+)?\b', comment)
        cleaned_words = []
        for word in words:
            if word not in combined_tokens:
                if ' ' in word:
                    cleaned_words.extend(word.lower().split())
                else:
                    cleaned_words.append(word.lower())
        #print('CLEANED WORDS:', cleaned_words)
        combined_tokens.extend(cleaned_words)
        #combined_tokens = set(combined_tokens)
        #print('Combined tokens:', combined_tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in combined_tokens if token not in stop_words]
        #print('all tokens:', lemmatized_tokens)
        comment_vectors = [w2vModel[token] for token in lemmatized_tokens if token in w2vModel]
        x_train.append(np.mean(comment_vectors, axis = 0) if comment_vectors else np.zeros(w2vModel.vector_size))

    x_train = np.array(x_train)

    y = np.array(train_class)

    le = LabelEncoder()
    y_train = le.fit_transform(y)

    Scaler = MinMaxScaler()
    x_train_scaled = Scaler.fit_transform(x_train)

    lr_w2v = lr.fit(x_train_scaled, y_train)
    return lr_w2v, x_train_scaled


# In[31]:


def classifySubreddit_test(text):
    """
    with open(text,  encoding='utf-8') as json_file:
        test_set = [json.loads(line) for line in json_file]

    subreddit_class_test = [line['subreddit'] for line in test_set ]
    comments_test = [line['body'] for line in test_set]
    """
    x_test = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    url_pattern = r'\((https?://[^\s]+)\)'
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    #replacements = lambda text: ''.join(replacement_rules.get(char, char) for char in text)
    #comments = list(map(replacements, comments))
    #print('\nnew comment line:', comment)
    comment = text
    for char, replacement in replacement_rules.items():
        comment = comment.replace(char, replacement)
    #for word in words:
    #if re.match(r'http[s]?://', comment):
    #Extract urls and add to token list. Text has useful urls for training
    combined_tokens = []
    urls = re.findall(url_pattern, comment)
    for url in urls:
        comment = re.sub(r'https?:\/\/(www\.)?', ' ', comment)
        sub_domain = url.split('/')
        domain_name = sub_domain[0]
        path_words = sub_domain[1:] if len(sub_domain) > 1 else []
        #Split domain and rest of url. Domain will likely be class specific
        split_domain = domain_name.split('.')
        path_tokens = []
        for i in path_words:
            path_tokens.extend(re.split(r'[\/\-_+\)\(\.]', i))            
        split_url = split_domain + [token for token in path_tokens if token.strip()]
        #Take care of edge cases
        for j in split_url:
            if ' ' in j:
                combined_tokens.extend(j.lower().split())
            else:
                combined_tokens.append(j.lower())
        #Remove stray punctuation
        for word in combined_tokens:
            word = re.sub(r'\W', '', word.lower())
    #Remove processed urls or else duplicates
    comment = re.sub(url_pattern, '', comment)
    words = re.findall(r'\b\w+(?:\'\w+)?\b', comment)
    cleaned_words = []
    for word in words:
        if word not in combined_tokens:
            if ' ' in word:
                cleaned_words.extend(word.lower().split())
            else:
                cleaned_words.append(word.lower())
    combined_tokens.extend(cleaned_words)
    #combined_tokens = set(combined_tokens)
    #print('Combined tokens:', combined_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in combined_tokens if token not in stop_words]
    #print('all tokens:', lemmatized_tokens)
    comment_vectors = [w2vModel[token] for token in lemmatized_tokens if token in w2vModel]
    x_test.append(np.mean(comment_vectors, axis = 0) if comment_vectors else np.zeros(w2vModel.vector_size))

    x_test = np.array(x_test)
    x_test_scaled = Scaler.transform(x_test)

    prediction = lr_w2v.predict(x_test_scaled)
    #lr_w2v.predict_proba(x_test)

    predicted_class = le.inverse_transform(prediction)

    return predicted_class[0]


# In[32]: