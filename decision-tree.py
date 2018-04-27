from sklearn.feature_selection import SelectPercentile, f_classif
import pickle
import numpy as np
import pandas as pd
np.random.seed(42)
from time import time



########################Load Data#########################
### The words (features) and authors (labels), already largely processed.
words_file = "word_data.pkl" 
authors_file = "from_data.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



####################Splitting Data########################
### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.2, random_state=42)
print len(features_train), len(features_test)



###############Creating TfidfVectorizer###################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
vocabulary = vectorizer.get_feature_names()


