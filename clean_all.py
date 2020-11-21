import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs_train(directory):
	documents = list()
	for filename in listdir(directory):
		if filename.startswith('.DS'):
			continue
		path = directory + '/' + filename
		doc = load_doc(path)
		tokens = clean_doc(doc)
		documents.append(tokens)
	return documents

# load and clean a dataset
def load_clean_dataset_train():
	P = process_docs_train('txt_int/DEM')
	H = process_docs_train('txt_int/HC')
	docs = P + H
	labels = array([0 for _ in range(len(P))] + [1 for _ in range(len(H))])
	return docs, labels


def process_docs_test(directory):
	documents = list()
	for filename in listdir(directory):
		if filename.startswith('.DS'):
			continue
		path = directory + '/' + filename
		doc = load_doc(path)
		tokens = clean_doc(doc)
		documents.append(tokens)
	return documents

def load_clean_dataset_test():
	P_T = process_docs_test('txt_test_int/DEM')
	H_T = process_docs_test('txt_test_int/HC')
	docs = P_T + H_T
	labels = array([0 for _ in range(len(P_T))] + [1 for _ in range(len(H_T))])
	return docs, labels

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load and clean all reviews
train_docs, ytrain = load_clean_dataset_train()
test_docs, ytest = load_clean_dataset_test()
# save training datasets
save_dataset([train_docs, ytrain], 'train.pkl')
save_dataset([test_docs, ytest], 'test.pkl')

