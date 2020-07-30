import os
import json
import pickle
import tensorflow as tf
from model import create_model

from eval_metric import EvalMetricReport
import sklearn.metrics as skm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--data_dir', type=str, default='data')

args = parser.parse_args()


def load_data(path):
    with open(path, 'rb') as f:
        inputs, labels = pickle.load(f)

    return [inputs['input_ids'], inputs['attention_mask']], labels


dir_list = os.listdir(args.result_dir)

for train_name in dir_list:
    # 모델 학습 파라미터 config 정보 load
    result_path = os.path.join(args.result_dir, train_name)
    with open(os.path.join(result_path, 'config.json'), 'r') as f:
        config = json.load(f)

    # test_set data load
    test_x, test_y = load_data(os.path.join(args.data_dir, config['test_set']))

    # model load
    model = create_model(None, config['max_seq_len'], config['num_classes'])
    model.load_weights(os.path.join(result_path, 'model.h5'))

    # predict
    outputs = model.predict(test_x, batch_size=config['batch_size'])
    y_pred = tf.cast(outputs >= 0.5, tf.int32).numpy()

    # save predict result
    with open(os.path.join(result_path, 'predict' + '.pkl'), 'wb') as f:
        pickle.dump(y_pred, f)

    # print Evaluation Metrics
    confusion_matrix = skm.multilabel_confusion_matrix(test_y, y_pred)

    print(train_name)
    report = EvalMetricReport(confusion_matrix)
    report.print_report()
    print('')