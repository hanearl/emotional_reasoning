import pickle
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split


def raw_data(path):
    with open(path, 'rb') as f:
        sentences, labels = pickle.load(f)
    return np.array(sentences).reshape([-1, 1]), np.array(labels)


def tokenize(tokenizer, sentences):
    encoding = tokenizer(sentences, return_tensors='pt',  padding='max_length', truncation=True, max_length=128)
    input_ids, token_type_ids, attention_mask = encoding['input_ids'], encoding['token_type_ids'], encoding[
        'attention_mask']
    return {
        'input_ids': np.array(input_ids),
        'token_type_ids': np.array(token_type_ids),
        'attention_mask': np.array(attention_mask)
    }


def main(tokenizer, raw_data, test_ratio, val_ratio):
    sentences, labels = raw_data

    x, y, x_test, y_test = iterative_train_test_split(sentences, labels, test_size=test_ratio)
    del sentences, labels
    x_train, y_train, x_val, y_val = iterative_train_test_split(x, y, test_size=val_ratio)

    x_test = [x[0] for x in x_test]
    with open('test_set.pkl', 'wb') as f:
        pickle.dump((tokenize(tokenizer, x_test), y_test), f)
    del x_test, y_test

    x_val = [x[0] for x in x_val]
    with open('val_set.pkl', 'wb') as f:
        pickle.dump((tokenize(tokenizer, x_val), y_val), f)
    del x_val, y_val

    x_train = [x[0] for x in x_train]
    with open('train_set.pkl', 'wb') as f:
        pickle.dump((tokenize(tokenizer, x_train), y_train), f)
    del x_train, y_train


# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# main(tokenizer=tokenizer,
#      raw_data=raw_data('/content/drive/My Drive/bert_sentiment/data/data.pkl'),
#      test_ratio=0.1,
#      val_ratio=0.1)