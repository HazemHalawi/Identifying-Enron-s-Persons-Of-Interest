import sys
from time import time
sys.path.append("../test/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




########################Training##########################
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#features_train = features_train[:len(features_train)/10]
#labels_train = labels_train[:len(labels_train)/10]

###The section below is used for Parameter tuning
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              #'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

clf = SVC(kernel="rbf", C=10000., gamma = 0.01)
t0 = time() #calculating training time
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#print("Best estimator found by grid search:")
#print(clf.best_estimator_)




###################Validating Model#######################
from sklearn.metrics import accuracy_score
t0 = time() #calculating testing time
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred,labels_test)
print accuracy



