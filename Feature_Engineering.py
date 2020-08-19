import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Load word2vec model
word2vec_model = Word2Vec.load("word2vec.model")


def plot_bar(df):
    number_of_texts = []
    for intent_name in df["intent"].unique():
        texts = df[df["intent"] == intent_name]["intent"].count()
        number_of_texts.append(texts)

    plt.figure(figsize=(20, 10))
    plt.bar([1, 3, 5, 7, 9, 11, 13], number_of_texts)
    plt.xlabel("Intent")
    plt.ylabel("Number of texts")
    plt.xticks([1, 3, 5, 7, 9, 11, 13], df["intent"].unique())
    plt.show()


def pre_process(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    return text


def label_data(df):
    category_codes = {
        'AddToPlaylist': 0,
        'BookRestaurant': 1,
        'GetWeather': 2,
        'PlayMusic': 3,
        'RateBook': 4,
        'SearchCreativeWork': 5,
        'SearchScreeningEvent': 6
    }
    df['category_code'] = df['intent']
    df = df.replace({'category_code': category_codes})
    return df


def convert_text_to_vector(df):
    tfidf = TfidfVectorizer()  # default params
    features = tfidf.fit_transform(df["text_clean"]).toarray()
    labels = df['category_code'].values
    return features, labels


def average_vectors_text(text):
    text_matrix = []

    for word in text.split(" "):
        text_matrix.append(word2vec_model.wv['word'])

    text_array = np.array(text_matrix)
    text_vector = LA.norm(text_array, axis=0)
    return text_vector
