from random import uniform
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from Read_JSON import *
from Feature_Engineering import *
from Word2Vec__Model import *

import numpy as np
import numpy.linalg as LA
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

# pre_process for training

features_train = train_df["text_clean"].apply(average_vectors_text)
features_test = test_df["text_clean"].apply(average_vectors_text)

features_train = features_train.tolist()
features_test = features_test.tolist()

with open("features_train_word2vec.pickle", "wb") as output:
    pickle.dump(features_train, output)

with open('features_test_word2vec.pickle', "wb") as output:
    pickle.dump(features_test, output)