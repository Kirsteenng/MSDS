#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:28:58 2022

@author: Kirsteenng
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import re
import os
import itertools

# Various paths for raw .txt files, insert paths here to reproduce code.
pos_words_path = '/Users/Kirsteenng_1/Desktop/UW courses/Winter 2022/CSE 517/Assignment 1/A1/opinion_lexicon_English/positive-words.txt'
neg_words_path = '/Users/Kirsteenng_1/Desktop/UW courses/Winter 2022/CSE 517/Assignment 1/A1/opinion_lexicon_English/negative-words.txt'
raw_data_pos = '/Users/Kirsteenng_1/Desktop/UW courses/Winter 2022/CSE 517/Assignment 1/A1/review_polarity/txt_sentoken/pos'
raw_data_neg = '/Users/Kirsteenng_1/Desktop/UW courses/Winter 2022/CSE 517/Assignment 1/A1/review_polarity/txt_sentoken/neg'

# Q1

def read_files(path):
    result_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                result_list.append(text)
            f.close()
    return result_list

# Produce a word: frequency dictionary
word_count = {}
def Word_count():
    for j in range(0,len(df['Text'])):    
        for i in df['Text'][j]:
            try:
                word_count[i] += 1
            except KeyError:
                word_count[i] = 1
        

# Find matching words in positive and negative set
def find_positive(text):
    pos_sum = 0
    for i in text:
        #print(i)
        if i in pos_words:
            pos_sum += 1
            #print(i)
    #print(pos_sum)
    return pos_sum
    
def find_negative(text):
    neg_sum = 0
    for i in text:
        if i in neg_words:
            #print(i)
            neg_sum += 1
    #print(neg_sum)
    return neg_sum
    
# Classify positive or negative for wach text
def classification(text): #individual list of tokens
    pos = find_positive(text)
    neg = find_negative(text)
    if pos > neg:
        return 1
    else:
        return-1
    

# Confusion matrix to produce TN,TP, FN, FP
def confusion_mat(df):
    TP,TN,FP,FN = 0,0,0,0
    for i in range(0,len(df)):
        
        label = df.iloc[i]['Label']
        classify = df.iloc[i]['Classification']
        if label == 1:
            if label != classify:
                FP += 1
                continue
            TP += 1
        else:
             if label != classify:
                 FN += 1
                 continue
             TN += 1
            
    return TP, TN, FP, FN

with open(pos_words_path,'r',encoding='utf-8',errors='ignore') as f:
    pos_words = f.read().splitlines()
    f.close()


with open(neg_words_path,'r',encoding='utf-8',errors='ignore') as f:
    neg_words = f.read().splitlines()
    f.close()
    

pos_list = read_files(raw_data_pos)
neg_list = read_files(raw_data_neg)

# Remove unnecessary words from dictionary
pos_words = pos_words[30:]
neg_words = neg_words[31:]

# Turn pos_list and neg_list into dataframes, add positive and negative labels
pos_df = pd.DataFrame(pos_list,columns = ['Text'])
neg_df = pd.DataFrame(neg_list,columns = ['Text'])

pos_df['Label'] = 1
neg_df['Label'] = -1

df = pos_df.append(neg_df).reset_index(drop = True)
stop_words = ['the','is','am','a','an','there','that','of','i','he','she','it','this',
'that','these','those','with','about','against','between','into','to','from','down',
'in','out','on','off','over','under','as',
'and','most','have','be','only','then','so','if','just','or','at','if']

# Data preprocessing: remove punctuation and tokenize set
df['Text'] = df['Text'].apply(lambda x:re.sub(r'[^\w\s]', '',x))
df['Text'] = df['Text'].apply(lambda x:x.strip().split())


# Classify the text based on frequency of lexicons
df['Classification'] = 0
for i in range(0,len(df)):
    df.at[i,'Classification'] = classification(df['Text'][i]) # using at will change the actual data rather than view
     
    
# Calculate accuracy and F1, see pdf for result
TP, TN, FP, FN = confusion_mat(df)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print('Accuracy for lexicon classifier = ', (TP+TN)/2000)
print('F1 Score for lexicon classifier(3dp) = ', round(2*precision*recall/(precision+recall),3) )



# ************** Logistic Regression *******************
# Separate into test and training set
  
indices = np.array(range(df.shape[0]))
num_training = int(0.8 * df.shape[0]) # 400 holdout for test data
np.random.shuffle(indices)
train_indices = indices[:num_training]
test_indices = indices[num_training:]

training_set = df.iloc[train_indices]
test_set = df.iloc[test_indices]

# Implement frequency feature model
def feature(text,dict_list):
    # Create a dictionary to keep track of the unique vocab in the corpus
    index_word = {}
    i = 0
    for word in dict_list:
        index_word[word] = i
        i += 1
        
    # Create a dictionary to keep count of word frequency
    count_dict = defaultdict(int)
    vec = np.zeros(len(dict_list))
    
    for item in text:
        try:
            #print(item)
            count_dict[item] += 1
        except KeyError:
            pass
        
    for key,item in count_dict.items():
        try:
            vec[index_word[key]] = item
        except KeyError:
            pass
    
    return vec
 

def sigmoid(score):
    sig = 1/(1+np.exp(-score))
    return sig

def SGD(theta,classify,feature):
    size = len(classify)
    gradient = 0
    for i in range(0,size):
        scalar = 1-sigmoid(classify[i] * np.dot(theta,feature[i]))
        gradient += scalar * -classify[i] * feature[i] # gradient will be a vector
        
    return gradient

# Calculate log loss for each theta update
def logloss(theta,classify,feature):
    size = len(classify)
    log_loss = 0
    for i in range (0,size):
        log_loss += np.log(1 + np.exp(-classify[i] * np.dot(feature[i],theta)))
        
    return log_loss

# Getting random sample for each epoch. 
# Input: df = training set df, percentage = percentage of sample size eg 20% should be inputted as 0.2
def random_sample(df,percentage):
    sample_size = int(len(df)*percentage)
    selected = df.sample(sample_size)
    x_train = selected['Feature']
    y_train = selected['Label']
    x_train = x_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    
    return x_train,y_train

def training(epoch,training_set,theta,loss_rec,alpha,x_test,y_test,acc_rec):
    #start = tm.time()
    for e in range(epoch):
        print('Entering epoch: ',e)
        x_train, y_train = random_sample(training_set,0.2)
        gradient = SGD(theta,y_train,x_train) # should be a vector, same length as theta
            
        #update theta
        theta = theta - alpha * gradient
        diff = logloss(theta,y_train,x_train) # y_train = vector of true label,x_train = vector of features
        loss_rec.append(diff)
        
        # insert y_pred, use theta to predict y_pred at each epoch
        # Predict using trained theta 
        y_pred = []    
        for j in range(0,len(x_test)):
            y_pred.append(np.sign(np.dot(theta,x_test['Feature'][j])))
        y_test_local = y_test
        y_test_local['Classification'] = y_pred
        LR_TP, LR_TN, LR_FP, LR_FN = confusion_mat(y_test_local)
        acc_rec.append([LR_TP, LR_TN, LR_FP, LR_FN])
    
        
    #end = tm.time()
    #print('Time spent on traning: ', end - start)
    return theta

    
# ****** Main script ******** # 
#bias = np.random.uniform(low=0, high=1, size=(1,))


# Create unique word list from corpus
word_list = list(itertools.chain.from_iterable(training_set.Text))
word_list = pd.DataFrame(word_list, columns = ['Text'])
word_list = word_list.Text.unique()
word_list = list(set(word_list) - set(stop_words)) #size = 43082

#training_set = training_set.drop(columns = ['Classification'],axis = 1)


# Create feature vector for each text in training set
m = len(training_set)
x_train_feature = []
for i in range(0,m):
    x_train_feature.append(feature(training_set.iloc[i]['Text'],word_list))
training_set['Feature'] = x_train_feature


#### Testing Logistic Regression ##
x_test = test_set['Text']
y_test = test_set['Label']
x_test = pd.DataFrame(x_test).reset_index(drop = True)
y_test = pd.DataFrame(y_test).reset_index(drop = True)


# Create feature vector for each text in test set
n = len(x_test)
x_test_feature =[]

for i in range(0,n):
    x_test_feature.append(feature(x_test['Text'][i],word_list))
x_test['Feature'] = x_test_feature # x_test is a df with columns Text, Feature

####   Start training ####
epoch = 100
alpha = 0.01
theta = np.random.uniform(low=-0.01, high=0.01, size=(len(word_list))) #same size as feature
loss_rec = []
accuracy_rec = []
trained_theta = training(epoch,training_set,theta,loss_rec,alpha,x_test,y_test,accuracy_rec)

# Plot accuracy graph
agg_acc = []
for i in range(0,len(accuracy_rec)):
    TP = accuracy_rec[i][0]
    TN = accuracy_rec[i][1]
    agg_acc.append((TP+TN)/400)
pd.DataFrame(agg_acc).plot()


# Plot error graph
pd.DataFrame(loss_rec).plot()

# Predict using trained theta 
#y_pred = []    
#for j in range(0,len(x_test)):
#    y_pred.append(np.sign(np.dot(trained_theta,x_test['Feature'][j])))
    
# Construct dataframe with Label and Classification
#y_test['Classification'] = [int(i) for i in y_pred]


# Calculate accuracy and F1 score
LR_TP, LR_TN, LR_FP, LR_FN = confusion_mat(y_test)
LR_precision = LR_TP/(LR_TP + LR_FP)
LR_recall = LR_TP/(LR_TP + LR_FN)
print('Accuracy for logistic classifier = ', (LR_TP + LR_TN)/400)
print('F1 Score for logistic classifier(3dp) = ', round(2 * LR_precision * LR_recall/(LR_precision + LR_recall),3) )


# Lexicon analysis
word_list[np.argmin(trained_theta)]
word_list[np.argmax(trained_theta)]

dict_list = word_list
pd.DataFrame(dict_list,columns = ['Vocabs'])
dict_list = pd.DataFrame(dict_list,columns = ['Vocabs'])
dict_list['Weight'] = trained_theta

dict_list.sort_values(by = 'Weight',ascending = False).head(20)
dict_list.sort_values(by = 'Weight',ascending = True).head(20)


def LR_fit(theta,text):
    feat = feature(text,word_list)
    return np.sign(np.dot(trained_theta,feat))
    

test1 = 'One of the movies ‘The Good, the Bad, and the Ugly’ is very amusing and appealing. I find nothing particularly stupid. And it should be more widely appreciated!'
classification(test1.split())
LR_fit(trained_theta, test1.split())

test2 = 'The alienated master is a perfectly great, sweet and fun person who is just allergic to false allegation and ambiguity.'
classification(test2.split())
LR_fit(trained_theta, test2.split())

test3 = 'We all think the suspect behind bar is a bad, awful and terrible person, given the absurdity and how angry, unpredictable he always is. But that is because he is being alienated all the time.'
classification(test3.split())
LR_fit(trained_theta, test3.split())


