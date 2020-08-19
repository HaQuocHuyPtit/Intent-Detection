import numpy as np
import sys
import os
import json
import pandas as pd
import pickle
from sklearn.utils import shuffle


def get_json_examples(intent_name, json_data):
    examples = []
    for e in json_data:
        parts = []
        for p in e['data']:
            parts.append(p['text'])
            examples.append({
                "intent": intent_name,
                "text": "".join(parts)
            })
    return examples


def create_df():
    train_json_examples = []
    test_json_examples = []

    for intent_name in next(os.walk("./data"))[1]:
        train_file = f"data/{intent_name}/train_{intent_name}_full.json"
        test_file = f"data/{intent_name}/validate_{intent_name}.json"

        with open(train_file, "r", encoding="utf8") as f:
            train_json = json.load(f)
            train_json_examples += get_json_examples(intent_name, train_json[intent_name])

        with open(test_file, "r", encoding="utf8") as f:
            test_json = json.load(f)
            test_json_examples += get_json_examples(intent_name, test_json[intent_name])

    train_df = pd.DataFrame.from_records(train_json_examples)
    test_df = pd.DataFrame.from_records(test_json_examples)

    train_df = shuffle(train_df, random_state=10)
    test_df = shuffle(test_df, random_state=10)

    with open("train_df.pickle", "wb") as output:
        pickle.dump(train_df, output)

    with open("test_df.pickle", "wb") as output:
        pickle.dump(test_df, output)


# Xu ly file pickle
create_df()
