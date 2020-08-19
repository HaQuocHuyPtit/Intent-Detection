from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from Feature_Engineering import *

import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load data
with open("train_df.pickle", "rb") as output:
    train_df = pickle.load(output)

print(train_df.head(5))
