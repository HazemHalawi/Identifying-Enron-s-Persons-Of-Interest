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





