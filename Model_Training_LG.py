from random import uniform
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Read_JSON import *
from Feature_Engineering import *

import numpy as np
import sys
import os
import json
import pickle
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

with open("train_df.pickle", "rb") as output:
    train_df = pickle.load(output)

with open("test_df.pickle", "rb") as output:
    test_df = pickle.load(output)

train_df["text_clean"] = train_df["text"].apply(pre_process)
test_df["text_clean"] = test_df["text"].apply(pre_process)

# data for training
train_df = label_data(train_df)
test_df = label_data(test_df)

# convert texts to vectors
tfidf = TfidfVectorizer()
features_train = tfidf.fit_transform(train_df['text_clean'])
features_test = tfidf.transform(test_df['text_clean'])

labels_train = train_df["intent"].values
labels_test = test_df["intent"].values

# train model
model = LogisticRegression()
model.fit(features_train, labels_train)

# check model
print(accuracy_score(labels_train, model.predict(features_train)))

print(accuracy_score(labels_test, model.predict(features_test)))
