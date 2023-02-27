import copy
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(file_name):
    print("starting to read: " + file_name)
    data3way = pd.read_json(file_name, lines=True)
    data3way = data3way[data3way["gold_label"].str.contains("-") == False]

    data3way['sentence1'] = data3way['sentence1'].str.replace('[^\w\s]', '')
    data3way['sentence1'] = data3way['sentence1'].str.lower()
    data3way['sentence1'] = data3way['sentence1'].values.tolist()

    data3way['sentence2'] = data3way['sentence2'].str.replace('[^\w\s]', '')
    data3way['sentence2'] = data3way['sentence2'].str.lower()
    data3way['sentence2'] = data3way['sentence2'].values.tolist()

    data2way = copy.deepcopy(data3way)
    data2way['gold_label'] = data2way['gold_label'].str.replace('neutral', 'contradiction')
    data3_labels = data3way['gold_label']
    data2_labels = data2way['gold_label']

    return data3way, data2way, data3_labels, data2_labels


def extract_features(file_name):
    sentence3way, sentence2way, label3way, label2way = read_data(file_name)

    sentence3way = sentence3way[['sentence1', 'sentence2']]
    sentence3way = sentence3way.apply(lambda x: ' , '.join(x.astype(str)), axis=1) #concat the sentences

    sentence2way = sentence2way[['sentence1', 'sentence2']]
    sentence2way = sentence2way.apply(lambda x: ' , '.join(x.astype(str)), axis=1)

    return sentence3way, sentence2way, label3way, label2way


def make_model(data, model_name):
    print("creating model for " + model_name)
    data_train = pd.DataFrame({'text': data})
    corpus_train = [row.split(' ') for row in data_train['text']]
    model = Word2Vec(sentences=corpus_train, vector_size=300, window=2, min_count=1, workers=4)
    model.save(model_name)

    return model


def evaluate(predicted, labels):
    acc = np.mean(predicted == labels)
    print("accuracy is:" + str(acc))


def create_vectors(data, model_train, name):
    print("creating vectors for " + name)
    data_vec = pd.DataFrame({'text': data})
    vectors = []
    for row in data_vec['text']:
        sentence1 = 0
        sentence2 = 0
        flag = 0
        for word in row.split():
            vec = 0
            if ',' == word: #seperator of the two sentences
                flag = 1
            elif word in model_train.wv.key_to_index:
                vec = np.array(model_train.wv[word])
            else:
                vec = np.zeros(shape=(300,))
            if flag:
                sentence2 += vec
            else:
                sentence1 += vec
        sentence = np.concatenate((sentence1,sentence2))
        vectors.append(sentence)
    return vectors


def prediction(vector_train_data, vector_test_data, train_labels):
    logistic_reg = LogisticRegression(max_iter=300).fit(vector_train_data, train_labels)
    predicted = logistic_reg.predict(vector_test_data)
    return predicted


def predict_and_eval(vector_train_data, vector_test_data, train_labels, test_labels):
    predicted = prediction(vector_train_data, vector_test_data, train_labels)
    evaluate(predicted, test_labels)

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    train3way_data, train2way_data, train3way_labels, train2way_labels = extract_features(config['train_data'])
    test3way_data, test2way_data, test3way_labels, test2way_labels = extract_features(config['test_data'])

    model3way = make_model(train3way_data, '3 way')
    model2way = make_model(train2way_data, '2 way')

    vector_3train_data = create_vectors(train3way_data, model3way, '3 way train set')
    vector_3test_data = create_vectors(test3way_data, model3way, '3 way test set')

    vector_2train_data = create_vectors(train2way_data, model2way, '2 way train set')
    vector_2test_data = create_vectors(test2way_data, model2way, '2 way test set')

    print("starting prediction and evaluation for 3 way classification")
    predict_and_eval(vector_3train_data, vector_3test_data, train3way_labels, test3way_labels)
    print("starting prediction and evaluation for 2 way classification")
    predict_and_eval(vector_2train_data, vector_2test_data, train2way_labels, test2way_labels)