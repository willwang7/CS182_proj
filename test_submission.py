import json, sys
import tensorflow

from sklearn import metrics,preprocessing,model_selection
from sklearn.metrics import accuracy_score
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import pandas as pd
import re
import spacy
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en import English

nltk.download('stopwords')
spacy.load('en')
parser = English()

import tensorflow_hub as hub
import tensorflow as tf

embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)) 
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]

def tokenizeText(text):
    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    
    tokens = parser(text)
    
    # lemmatization
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    
    # reomve stop words and special charaters
    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    
    tokens = [tok for tok in tokens if len(tok) >= 3]
    
    # remove remaining tokens that are not alphabetic
    tokens = [tok for tok in tokens if tok.isalpha()]
    
    tokens = list(set(tokens))
    
    return ' '.join(tokens[:])


def eval(text):
	text_list = tokenizeText(text)
	x_test = text_list.tolist()
	input_text = Input(shape=(1,), dtype=tf.string)
	embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
	dense = Dense(256, activation='relu')(embedding)
	middle = Dense(128, activation='relu')(dense)
	pred = Dense(5, activation='softmax')(middle)
	model = Model(inputs=[input_text], outputs=pred)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	with tf.Session() as session:
	    K.set_session(session)
	    session.run(tf.global_variables_initializer())
	    session.run(tf.tables_initializer())
	    model.load_weights('./response-elmo-model.h5')  
	    predicts = model.predict(x_test, batch_size=4)
	return predicts

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")