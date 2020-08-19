from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from Feature_Engineering import *

import pickle

# Read root file
with open("train_df.pickle", "rb") as data:
    train_df = pickle.load(data)

with open("test_df.pickle", "rb") as data:
    test_df = pickle.load(data)

# Pre_process for word2vec model
train_df["text_clean"] = train_df["text"].apply(pre_process)
test_df["text_clean"] = test_df["text"].apply(pre_process)

sentences = []

for sentence in train_df["text_clean"].values:
    tokenized = []
    for word in sentence.split(" "):
        tokenized.append(word)
    sentences.append(tokenized)

# Training word2vec model
path = get_tmpfile("word2vec.model")

model = Word2Vec(sentences, size=100, window=5, min_count=1, sg=1, workers=4)  # using skipGram
model.save("word2vec.model")

