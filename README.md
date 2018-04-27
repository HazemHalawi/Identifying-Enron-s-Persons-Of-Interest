# MachineLearning

The Enron Email Dataset contains a total of about 0.5M email messages from about 150 users, mostly senior management of Enron, the company that had the largest bankruptcy reorganization in American history at that time, due to fraud.

The goal of this project is to build machine learning model that can identify persons of interest (poi: which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.) based on Enron's email dataset.

Machine learning can identify patterns in huge datasets, in this case the Enron dataset including Enron's employees’ financial data and all the emails sent between them, and build a analytical model based on it. And so by analysing the emails, it can discover patterns for the two labelled classes 'poi' for emails sent by a poi and 'non-poi' for emails sent by a non-poi. Thus, in later cases for any email for a specific person we can classify him as a one of the two classes.

I downloaded the dataset from : https://www.cs.cmu.edu/~enron/

The [emails-processing.py](https://github.com/HazemHalawi/MachineLearning/blob/master/emails-processing.py) file, takes the emails with a list of poi names, and creates tow files. The first file (word_data.pkl) contains the features (emails) and the second (from_data.pkl) contain the asoociated labels (POI or NonPOI). Both file are in the Features-Labels.zip. This algorithm also does some text learning for each email.

I tried three different algorithms when building my machine learning model (decision tree, Naive Bayes and SVM).



