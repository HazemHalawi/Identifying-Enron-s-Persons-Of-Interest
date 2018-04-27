import sys
import numpy as np
from time import time
sys.path.append("../test/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



########################Training##########################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
t0 = time() #calculating training time
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"



###################Validating Model#######################
from sklearn.metrics import accuracy_score
t0 = time() #calculating testing time
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred,labels_test)
print accuracy








