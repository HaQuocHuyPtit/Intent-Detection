import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# load root file
with open("train_df.pickle", "rb") as data:
    train_df = pickle.load(data)

with open("test_df.pickle", "rb") as data:
    test_df = pickle.load(data)

# load features file
with open("features_train_word2vec.pickle", "rb") as data:
    features_train_word2vec = pickle.load(data)

with open("features_test_word2vec.pickle", "rb") as data:
    features_test_word2vec = pickle.load(data)

# load labels file
with open("labels_train.pickle", "rb") as data:
    labels_train = pickle.load(data)

with open("labels_test.pickle", "rb") as data:
    labels_test = pickle.load(data)

# train based model
mlp = MLPClassifier()

mlp.fit(features_train_word2vec, labels_train)

# check based model
print("Accuracy on train_data: ")
print(accuracy_score(labels_train, mlp.predict(features_train_word2vec)))

print("Accuracy on test_data: ")
print(accuracy_score(labels_test, mlp.predict(features_test_word2vec)))