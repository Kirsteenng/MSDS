#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:57:06 2022

@author: Kirsteenng
"""

# Project Goal and Background
This project is to develop obfuscation methods where we are required to replace a set of pre-identified words in the text data 
in order to neutralise the accuracy of gender classification while retaining or improving the accuracy of genre classification of the given text data.

We will refer to these definitions in this project, 
target = the word that is to be obfuscated
substitute = word from opposite gender list

The goal is to identify target and replace it with substitute from the opposite gender list.


# Project Description
The training data under dataset.csv is a list of Reddit posts containing the text data, subreddit genre 
and the gender of the poster. Male.txt and female.txt are two documents containing a set of predefined words
that will be used as target(substitute) in our obfuscation tasks. 


Three methods have been developed for obfuscation. 
1. Random obfuscation: randonly replace target with substitute from opposite gender list.
2. Maximum Cosine Similarity obfuscation: select the substitute by calculating the maximum cosine similarity
based on gloVe embedding vectors.
3. Lottery obfuscation: select 1/3 options (random,maximum cosine similarity,no replacement)

# Files required
1. dataset.csv
2. male.txt
3. female.txt
4. glove.6B.50d.tx #download from [here](https://nlp.stanford.edu/projects/glove/)) 