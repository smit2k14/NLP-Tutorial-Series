import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Creating the vectorizer for the raw text
text_vectorizer = TfidfVectorizer(
	tokenizer = None,
	preprocessor = None,
	decode_error = 'replace',
	stop_words = None,
	ngram_range = (1,3),
	max_features=10000,
	min_df=5,
	max_df=0.75,
	norm = None
	)

#Creating the vectorizer for the Parts of Speech of the text
pos_vectorizer = TfidfVectorizer(
	tokenizer=None,
	lowercase=False,
	preprocessor=None,
	ngram_range=(1, 3),
	stop_words=None,
	use_idf=False,
	smooth_idf=False,
	norm=None,
	decode_error='replace',
	max_features=5000,
	min_df=5,
	max_df=0.75,
	)


raw_text_data = []
labels = []
with open('imdb_labelled.txt') as f:
	for line in f:
		raw_text, label = line.split('\t')
		raw_text_data.append(raw_text[:-2])
		labels.append(int(label[0]))

'''
Preprocessing the data
Part 1. Removing Stop Words, Converting to lowercase
Part 2. Removing Special Characters
Part 3. Lemmatizing the data
'''
def remove_stopwords(text):
	words = word_tokenize(text)
	output_text = []
	for w in words:
		if w not in stop_words:
			output_text.append(w)
	output_text = ' '.join(output_text).replace(' , ',',').replace(' .','.').replace(' !','!')
	output_text = output_text.replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')   
	return output_text


def preprocessing_data(raw_data, lemmatizer):
	preprocessed_data = []
	for data in raw_data:
		data = data.lower()
		data = remove_stopwords(data)
		data = lemmatizer.lemmatize(data)
		data = ' '.join(re.sub("(@[_A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
		preprocessed_data.append(data)
	return preprocessed_data


'''
Getting the POS for the raw text. pos() gets the POS for the sentence and pos_data() gets it for the entire text.
'''
def pos(data):
	return nltk.pos_tag(data)

def pos_data(pre_data):
	posData = []
	for d in pre_data:  
		pdata = pos(d.split())[:]
		for i in range(len(pdata)):
				pdata[i] = pdata[i][1]
		posData.append(' '.join(pdata))
	return posData

pre_data = preprocessing_data(raw_text_data, lemmatizer)

pos_pre_data = pos_data(pre_data)

pre_data = text_vectorizer.fit_transform(pd.Series(pre_data)).toarray()
pos_data = pos_vectorizer.fit_transform(pd.Series(pos_pre_data)).toarray()

#Combining the features of both TFIDFs and splitting into train and test data
train_data = np.concatenate((pre_data, pos_data), axis = 1)
train_x, test_x, train_y, test_y = train_test_split(train_data, labels)

#Logistic Regression Classifer : Score 0.75
log_reg = LogisticRegression()
log_reg.fit(train_x, train_y)

print(log_reg.score(test_x, test_y))

#Decision Tree Classifer : Score 0.68
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x, train_y)

print(dec_tree.score(test_x, test_y))

#Scores may fluctuate due to random initialization of weights.