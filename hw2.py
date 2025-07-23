#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk
import json
import os
import re
import math


# In[8]:


#os.chdir('Downloads')


# In[436]:


#problem1_trainingFile = 'problem1_trainingFile.txt'


# In[439]:


"""
trainFile: a text file, where each line is arbitratry human-generated text
Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
"""
def calcNGrams_train(problem1_trainingFile):

    #with open(problem1_trainingFile, encoding='utf-8').read() as f:
        #sentences = f

    global unigram_prob
    global bigram_probabilities
    global vocab
    global word_count

    sentences = open(problem1_trainingFile, encoding='utf-8').read()

    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    for symbol, replacement in replacement_rules.items():
        sentences = sentences.replace(symbol, replacement)

    sentence_pattern = re.compile(r'(?<!["\'])\s*(?<=\.|\?|!)\s+')

    split_sentences = sentence_pattern.split(sentences)

    processed_sentences = []

    # Add start and end tokens to each sentence
    for sentence in split_sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            processed_sentences.append(f"<s> {stripped_sentence} </s>")

    reduced_text = []
    for sentence in processed_sentences:
        sentence = sentence.lower()
        reduced_text.append(sentence)
    #print(reduced_text)

    words = []
    for sentence in reduced_text:
        found_words = re.findall(r'<s>|\b\w+\b|</s>', sentence) 
        words += found_words
    vocab = set(words)

    #Add unknown to the vocabulary
    vocab.add('<UNK>')
    vocab_size = len(vocab)

    #Need total word count for probabilities
    total_words = len(words)

    #Create unigram count
    word_count =  {}
    for word in words:
        if word not in word_count:
            word_count[word] = 1
        else: 
            word_count[word] += 1

    unigram_prob = {word: count / total_words for word, count in word_count.items()}

    #Create Bigrams
    bi_grams=[]
    i=0
    while(i<len(words)):
            bi_grams.append(words[i:i+2])
            i=i+1
    bi_grams=bi_grams[:-1]

    #Create bigram counts
    bigram_counts = {}
    for i in range(len(words) - 1):
        word_1 = words[i]
        word_2 = words[i+1]
        #Loop to add word sets to bigram
        if bigram_counts == ('</s>', '<s>'):
            continue
        if word_1 not in bigram_counts:
            #Need to create new entry which will add 1
            bigram_counts[word_1] = {}
        if word_2 not in bigram_counts[word_1]:
            bigram_counts[word_1][word_2] = 1
        else:
            #increment
            bigram_counts[word_1][word_2] += 1
    #Create bigram probabilities
    bigram_probabilities = {}
    #Go through dictionary counts
    for i, j in bigram_counts.items():
        bigram_pairs = sum(j.values())
        bigram_probabilities[i] = {}
        for j, count in j.items():
            bigram_probabilities[i][j] = count / bigram_pairs
            
    #Return variables needed for test script
    return word_count, bigram_probabilities, unigram_prob, vocab


# In[440]:


#word_count, bigram_probabilities, unigram_prob, vocab = calcNGrams_train(problem1_trainingFile)


# In[441]:


"""
sentences: A list of single sentences. All but one of these consists of entirely random words.
Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
"""
def calcNGrams_test(sentences):
    vocab_size = len(vocab)
    #uniform_model = 1/vocab_size
    total_words = sum(word_count.values())
    #Create initial placeholder prob
    min_prob = math.log(1/(vocab_size * total_words))
    target = -1
    for index, sentence in enumerate(sentences):
        sentence = re.findall(r'\b\w+\b', sentence.lower())
        #Crreate initial log prob score
        prob_total = 0
        padded_sentence = ['<s>'] + [word if word in vocab else '<UNK>' for word in sentence] + ['</s>']
        #print(padded_sentence)
        for word in range(len(padded_sentence) - 1):
            word_1 = padded_sentence[word]
            word_2 = padded_sentence[word+1]
            if word_1 in bigram_probabilities and word_2 in bigram_probabilities[word_1]:
                prob = bigram_probabilities[word_1][word_2]
            elif word_2 in unigram_prob:
                prob = unigram_prob[word_2]
            else:
                prob = 1/(vocab_size * total_words)
            log_prob = math.log(prob)
            prob_total += log_prob
        #print(prob_total)
        #This should be in the inner loop or will reset every sentence
        if prob_total < min_prob:
            min_prob = prob_total
            target = index
    
    return target


# In[442]:


#calcNGrams_test(P)
#print(idx)


# In[4]:


#problem2_trainingFile = 'problem2_trainingFile.jsonlist'


# In[85]:


"""
trainFile: A jsonlist file, where each line is a json object. Each object contains:
    "review": A string which is the review of a movie
    "sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
"""
def calcSentiment_train(problem2_trainingFile):
    global nb_clf
    global vocab_nb
    reviews = []
    sentiments = []
    lemmatizer = WordNetLemmatizer()
    vocab_nb = set()
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    stop_words = set(stopwords.words('english'))
    
    with open(problem2_trainingFile, 'r') as json_file:
        for line in json_file:
            #print(line)
            json_data = json.loads(line)
            review = json_data['review']
            review = review.replace('-', ' ')
            for symbol, replacement in replacement_rules.items():
                review = review.replace(symbol, replacement)
            review = re.sub(r'[^a-z\s]', '', review.lower())
            tokens = review.split()
            tokens = [word for word in tokens if word not in stop_words]
            lemmatized_tokens = []
            negatives = False
            for i, word in enumerate(tokens):
                if word == "not" or word.endswith("n't"):
                    negatives = True
                    continue
                if negatives:
                    word = "NOT_" + word
                    negatives = False

            # Lemmatize each word based on its POS tag
                syn_tag = wordnet.synsets(word)[0].pos() if wordnet.synsets(word) else 'n'
                syn_tag = syn_tag if syn_tag in ['a', 'r', 'n', 'v'] else 'n'
                lemmatized_word = lemmatizer.lemmatize(word, syn_tag)
                lemmatized_tokens.append(lemmatized_word)
            word_set = set(lemmatized_tokens)
            vocab_nb.update(word_set)
            #stemmed_token = [ps.stem(word) for word in tokens]
            review = ' '.join(lemmatized_tokens)
            sentiment = 1 if json_data['sentiment']  else 0
            reviews.append(review)
            sentiments.append(sentiment)

    vocab_nb = list(vocab_nb)

    #print(sorted(vocab_nb))

    nbcounts = []
    vocabulary_dict = {word: index for index, word in enumerate(vocab_nb)}
    for review in reviews:
        vector = [0] * len(vocab_nb)
        #Set to count only once per occurrence per lecture
        words = set(review.split())
        for word in words:
            if word in vocabulary_dict:
                index = vocabulary_dict[word]
                vector[index] = 1
        nbcounts.append(vector)

    nb_clf = MultinomialNB(alpha = 1)
    nb_clf.fit(nbcounts, sentiments)

    return nb_clf, vocab_nb


# In[86]:


#nb_clf, vocab_nb = calcSentiment_train(problem2_trainingFile)


# In[87]:


"""
review: A string which is a review of a movie
Return a boolean which is the predicted sentiment of the review.
Must run in under 120 seconds, and must use Naive Bayes
"""
def calcSentiment_test(review):
    review = review.lower().replace('-', ' ')
    review = re.sub(r'[^a-z\s]', '', review)
    tokens = review.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    negatives = False
    for word in tokens:
        if word == "not" or word.endswith("n't"):
            negatives = True
            continue
        if negatives:
            word = "NOT_" + word
            negatives = False

        syn_tag = wordnet.synsets(word)[0].pos() if wordnet.synsets(word) else 'n'
        syn_tag = syn_tag if syn_tag in ['a', 'r', 'n', 'v'] else 'n'
        lemmatized_word = lemmatizer.lemmatize(word, syn_tag)
        lemmatized_tokens.append(lemmatized_word)

    # Vectorize the new review
    nbcounts = [0] * len(vocab_nb)
    tokenized_review = set(lemmatized_tokens)
    for word in tokenized_review:
        if word in vocab_nb:
            index = vocab_nb.index(word)
            nbcounts[index] = 1

    # Predict sentiment (True for positive, False for negative)
    prediction = nb_clf.predict([nbcounts])[0]
    return True if prediction == 1 else False


# In[92]:


#calcSentiment_test(problems2)


# In[38]:




