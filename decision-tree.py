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



#features_train = features_train[:150].toarray()
#labels_train   = labels_train[:150]
########################Training###########################
from sklearn import tree
from sklearn.model_selection import GridSearchCV

###The section below is used for Parameter tuning
#param_grid = {'min_samples_split': [2, 3, 4, 5, 10, 20, 40, 50, 100] }
#clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)

clf = tree.DecisionTreeClassifier()#min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#print("Best estimator found by grid search:")
#print(clf.best_estimator_)



###################Validating Model#######################
###We make predictions on the test set, and then we 
###calculate the accuracy the validate our model performance.
from sklearn.metrics import accuracy_score
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

###Calculating Accuracy
accuracy = accuracy_score(pred, labels_test)
print "Accuracy : ", accuracy



##############Evaluating Model Performance#################
###Using recall and percision
#pred = pred.tolist()
from sklearn.metrics import precision_score, recall_score, f1_score
labels_test = np.asarray(labels_test)
labels_test = labels_test.astype(np.int)
pred = pred.astype(np.int)
###Calculating Precision and recall
print "Precision:", precision_score(labels_test, pred)
print "Recall:", recall_score(labels_test, pred)
print "f1_score:", f1_score(labels_test, pred)



###################Features importance####################
###Get the most important feature, the feature's number 
###and the actual word.
feature_number = 0
for i in clf.feature_importances_:
	feature_number = feature_number +1
	if i > 0.01 :
		print 'feature number:', feature_number, ', feature importance:', i, ', Word: ', vocabulary[feature_number-1]
print len(vocabulary)





####################Terms Frequency########################
#import nltk
##nltk.download('punkt')
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#with open("../test/word_data.pkl", "r") as f:
    #data = f.read()
#words= word_tokenize(data)
#useful_words = [word  for word in words if word not in stopwords.words('english')]
#frequency = nltk.FreqDist(useful_words)
#print(frequency.most_common(100))
#-->[('enron', 21248), ('would', 9826), ('program', 9206), ('forward', 8847), ('david', 8774), ('compani', 8420), ('schedul', 8390), ('one', 8233), ('w', 8105), ('price', 7980), ('busi', 7900), ('pleas', 7821), ('time', 7465), ('new', 7399), ('subject', 7261), ('market', 7220), ('power', 6988), ('final', 6873), ('databasealia', 6729), ('ani', 6595), ('hour', 6528), ('need', 6520), ('deal', 6376), ('know', 6335), ('provid', 5912), ('also', 5908), ('meet', 5797), ('work', 5569), ('oper', 5543), ('manag', 5357), ('date', 5333), ('energi', 5183), ('like', 5097), ('pm', 5018), ('dbcaps97dataunknown', 4994), ('file', 4895), ('import', 4806), ('close', 4778), ('continu', 4740), ('said', 4738), ('get', 4675), ('group', 4673), ('trade', 4593), ('make', 4588), ('associateanalyst', 4584), ('may', 4556), ('hourahead', 4524), ('perform', 4360), ('us', 4332), ('call', 4189), ('peopl', 4182), ('pmto', 4135), ('year', 4124), ('2001', 4122), ('john', 4048), ('current', 4048), ('california', 3993), ('plan', 3897), ('issu', 3866), ('email', 3860), ('day', 3784), ('direct', 3779), ('gas', 3688), ('week', 3610), ('inform', 3599), ('mani', 3549), ('discuss', 3529), ('ena', 3512), ('mark', 3457), ('chang', 3412), ('let', 3400), ('remain', 3398), ('onli', 3379), ('delaineyhouect', 3293), ('includ', 3268), ('2', 3267), ('creat', 3265), ('report', 3212), ('cap', 3176), ('review', 3174), ('want', 3167), ('communic', 3165), ('go', 3110), ('process', 3101), ('use', 3101), ('month', 3098), ('unit', 3093), ('risk', 3061), ('next', 3037), ('varianc', 3036), ('question', 3022), ('term', 3011), ('opportun', 3010), ('jeff', 3010), ('dbcaps97data', 3005), ('success', 3001), ('requir', 2999), ('think', 2996), ('j', 2979), ('origin', 2970)]






