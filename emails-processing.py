import pickle
import sys
import os
import string
from nltk.stem.snowball import SnowballStemmer
from time import time

from email.parser import Parser
sys.path.append( "../final_project/" )

from poi_email_addresses import poiEmails
poi_email_list = poiEmails()
#print poi_email_list

 
def check_poi(email):
	if email in poi_email_list:
		return '1'
	else :
		return '0'



def email_analyse(inputfile, to_email_list, from_email_list, email_body, numPoi, numNonPoi):
    if os.path.exists(inputfile):
	data = open(inputfile, "r") 
	for path in data:
		path = os.path.join('../Enron_Mail', path[:-1])
		if os.path.exists(path):
			with open(path, "r") as f2:
				email_data = f2.read()
	    		email = Parser().parsestr(email_data)		
			###set a max number 
			poi = check_poi(email['from'])
			#This email has a lot of emails, so similar that they are driving the whole learning process
			if email['from'] != 'pete.davis@enron.com':
				if (poi == '0' and numNonPoi < 8800 ) or (poi == '1' and numPoi < 8800):
					if poi == '0':	
					 	numNonPoi = numNonPoi + 1
					elif poi == '1':
						numPoi = numPoi + 1
					#Add poi as boolean
		   			from_email_list.append(check_poi(email['from']))			
	
					##Add the email_to mail
					#if email['to']:
		        		#	email_to = email['to']
		       			#	email_to = email_to.replace("\n", "")
		        		#	email_to = email_to.replace("\t", "")
					#	email_to = email_to.replace("\r", "")
		        		#	email_to = email_to.replace(" ", "")
		        		#	email_to = email_to.split(",")
		        		#	for email_to_1 in email_to:
					#		#if all of them are pois them to_email is '1'
					#		to_email_list.append(email_to_1)
					print email['from']
	
					#Add the content
					content = email_data.split("X-FileName:")
					text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
		     			text_string = text_string.replace("\n", " ")
		        		text_string = text_string.replace("\t", " ")
					text_string = text_string.replace("\r", " ")
					stemmer = SnowballStemmer("english")
					words = ""
					for word in text_string.split( ):
						words = words + stemmer.stem(word) + " "
					### use str.replace() to remove any instances of the words
		        		remove = ["jmf","ddelainnsf","delainey","kitchen","ddelainnsf","2702pst","holden", "salisburi", 						"62602pst","nonprivilegedpst"]
					for word in remove: 
					    words = words.replace(word,"") 
		    			email_body.append(words)
		
	data.close()	
	return numPoi, numNonPoi
t0 = time()

enron_data = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
all_emails = []
for key,value in enron_data.iteritems():
	all_emails.append(enron_data[key]["email_address"])
#removing poi mails from all mails
nonpoi_email_list = [x for x in all_emails if x not in poi_email_list]
#removing NaN values
nonpoi_email_list = [x for x in nonpoi_email_list if x not in 'NaN']
#print nonpoi_email_list



to_data = []
from_data = []
word_data = []

i = 0
# numPoi and numNonPoi are counters for the max number of emails 
# for respectively a poi and a nonpoi
numPoi = 0
numNonPoi = 0
file_to_read = "/home/hazem/ud120-projects/final_project/emails_by_address"
for directory, subdirectory, filenames in  os.walk(file_to_read):
	#print(directory, subdirectory, len(filenames))
	print len(filenames)
	for filename in filenames:
		#this is a counter for the number of files to read
		if ( numPoi < 8800 ) or ( numNonPoi < 8800 )  :
			numPoi, numNonPoi = email_analyse(os.path.join(directory,filename), to_data, from_data, word_data, numPoi, numNonPoi )
			#i = i + 1
			print numPoi, numNonPoi

#pickle.dump( to_data, open("to_data_2.pkl", "w") )
pickle.dump( from_data, open("from_data.pkl", "w") )
pickle.dump( word_data, open("word_data.pkl", "w") )



print "training time:", round(time()-t0, 3), "s"











