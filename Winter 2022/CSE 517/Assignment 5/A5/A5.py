#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 21:47:57 2022

@author: Kirsteenng
"""

# Import libraries
import pandas as pd
import numpy as np
import random as rd
from scipy.spatial import distance

# Read file
def openfile(path:str)->pd.DataFrame:
    with open(path,'r',encoding = 'utf-8') as f:
        df = f.read().splitlines()
    return df

# Extract and replace words randomly from gender == M that appears in male_df
def extract_replace_random(data:pd.DataFrame,gender_dict:set)-> str:
    text = data['post_text'].split()
    #print(text)
    for i in range(0,len(text)):
        if text[i] in gender_dict[data['op_gender']]:
            #print('word in set: ',text[i])
            replacement = rd.choice(list(gender_dict[data['opposite_gender']]))
            #print('replacement word: ',replacement)
            text[i] = replacement
    return ' '.join(text)

# Generate a probability dictionary
def Gen_probabilityDict(text:list) -> dict:
    probability_dict ={}
    for i in range(0,len(text)):
        split_text = text[i].split()
        probability_dict[split_text[0]] = np.array(split_text[1:], dtype= np.float32)
        
    return probability_dict

# Turn set of words into a dictionary
def Vectorised(word_set:set,embedding:set)->dict:
    result = {}
    for word in list(word_set):
        try:
            result[word] = embedding[word]
        except KeyError:
            print('This doesnt exist in gloVe: ',word)
            result[word] = np.array([1]* 50)
    return result

# Return word that contains highest cosine similarity
def Find_maxCos(target_word:np.array, oppo_gender:dict) -> str:
    max_cos = 0
    for i in oppo_gender.keys():
        current_cos = 1 - distance.cosine(target_word,oppo_gender[i])
        if current_cos > max_cos:
            similar_word = i    
            max_cos = current_cos
    
    return similar_word

# Loop through the original text and replace target word
def Loop_replace(text:list,gender_dict:dict,opposite_dict:dict)-> str:
    for index in range(0,len(text)):
        if text[index] in gender_dict:
            #print('word in set: ',text[i])
            if np.any(gender_dict[text[index]]):
                replacement = Find_maxCos(gender_dict[text[index]],opposite_dict)
                text[index] = replacement
    return ' '.join(text)

# Extract and replace words with highest Cosine similiarity from opposite dictionary
def extract_replace_cosine(data:pd.DataFrame,female:dict,male:dict)-> str:
    text = data['post_text'].split()
    #print(text)
    if data['op_gender'] == 'M':
        print('this is male')
        join_text = Loop_replace(text,male_dict,fm_dict)
    else:
        print('********* this is female')
        join_text = Loop_replace(text,fm_dict,male_dict)
                
    return join_text

# Extract and replace words with based on 1/3 rzndomly selected option
def extract_replace_lottery(data:pd.DataFrame,female:dict,male:dict,gender_dict)-> str:
    text = data['post_text'].split()
    for index in range(0,len(text)):
        if text[index] in gender_dict[data['op_gender']]:
            option = rd.randint(1,3)
            #print('Option sampled: ',option)
            # Option 1: random replacement with opposite gender dict 
            if option == 1:
                #print('Subbing with random')
                replacement = rd.choice(list(gender_dict[data['opposite_gender']]))
                
            # Option 2: random replacement with maximum cosine
            elif option == 2:
                #print('Subbing with max cos')
                if data['op_gender'] == 'M':
                    replacement = Find_maxCos(male[text[index]],female)
                else:
                    replacement = Find_maxCos(female[text[index]],male)
            
            else:
                continue
            
            text[index] = replacement
        
    return ' '.join(text)
 
    
#### ___________________________________________ ####


# Paths
male_path = './male.txt'
female_path = './female.txt'
data_path = './dataset.csv'
embedding_path = './glove/glove.6B.50d.txt'

male_set = set(openfile(male_path))
fm_set = set(openfile(female_path))
embedding = openfile(embedding_path)

data_df = pd.read_csv(data_path)
data_df['opposite_gender'] = data_df['op_gender'].apply(lambda x: 'W' if x == 'M' else 'M')
gender_dict = {'M':male_set,'W':fm_set}

##### Random obfuscation
for i in range(0,len(data_df)):
    data_df.loc[i,'post_text'] = extract_replace_random(data_df.loc[i],gender_dict)

data_df.to_csv('test.csv')


#### Replacing words with max cosine similarity
embedding = Gen_probabilityDict(embedding)
replace_cosine = pd.read_csv(data_path)
replace_cosine['opposite_gender'] = replace_cosine['op_gender'].apply(lambda x: 'W' if x == 'M' else 'M')

# Get max cosine pair for each word in female and male.txt
fm_dict = Vectorised(fm_set, embedding)
male_dict = Vectorised(male_set, embedding)

for i in range(0,len(replace_cosine)):
    replace_cosine.loc[i,'post_text'] = extract_replace_cosine(replace_cosine.loc[i],fm_dict,male_dict)

replace_cosine.to_csv('cosine.csv')

# Randomly assigned 30% between cosine, random and no replacement
random_replace = pd.read_csv(data_path)
random_replace['opposite_gender'] = random_replace['op_gender'].apply(lambda x: 'W' if x == 'M' else 'M')

for i in range(0,len(random_replace)):
    random_replace.loc[i,'post_text'] = extract_replace_lottery(random_replace.loc[i],fm_dict,male_dict,gender_dict)

random_replace.to_csv('lottery.csv')
