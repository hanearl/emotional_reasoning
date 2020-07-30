import numpy as np
import json


class EvalMetric:
    def __init__(self, c_matrix):
        self.matrix = c_matrix

    @property
    def accuracy(self):
        return self.matrix[1][1] / (self.matrix[1][0] + self.matrix[1][1] + self.matrix[0][1] + 1e-8)

    @property
    def recall(self):
        return self.matrix[1][1] / (self.matrix[1][0] + self.matrix[1][1] + 1e-8)

    @property
    def precision(self):
        return self.matrix[1][1] / (self.matrix[0][1] + self.matrix[1][1] + 1e-8)

    @property
    def f1(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-8)

    @property
    def f05(self):
        return 1.25 * (self.precision * self.recall) / (0.25 * self.precision + self.recall + 1e-8)

    @property
    def f2(self):
        return 5 * (self.precision * self.recall) / (4 * self.precision + self.recall + 1e-8)

    def __getitem__(self, key):
        return getattr(self, key)

    def print_eval_metric(self):
        print('accuracy\t%0.4f\tf1\t%0.4f\tprecision\t%0.4f\trecall\t%0.4f' %
              (self.accuracy*100, self.f1*100, self.precision*100, self.recall*100))
        print('')


class EvalMetricReport:
    def __init__(self, c_matrix_list):
        self.c_matrix_list = c_matrix_list
        with open('data/info.json', 'r') as f:
            info = json.load(f)
            self.labels = info['labels']
            self.groups = info['groups']

        self.group_to_label = {
            1: [i for i, group in enumerate(self.groups) if group == 1],
            2: [i for i, group in enumerate(self.groups) if group == 2],
            3: [i for i, group in enumerate(self.groups) if group == 3]
        }

        # 모든 Class Metric
        self.c_matrix = np.array(self.c_matrix_list).sum(axis=0)
        self.all_metric = EvalMetric(self.c_matrix)

        # 중립 Class 제거 Metric
        self.non_neu_c_matrix = np.delete(self.c_matrix_list, 3, axis=0).sum(axis=0)
        self.non_neu_metric = EvalMetric(self.non_neu_c_matrix)

        # 그룹 별 Metric
        self.group_metric = self.get_group_metric()

    def __getitem__(self, key):
        return getattr(self, key)

    def get_group_metric(self):
        c_matrix = self.c_matrix_list

        def group_eval_metric(group_list):
            nonlocal c_matrix
            group_c_matrix = c_matrix[group_list].sum(axis=0)
            return EvalMetric(group_c_matrix)

        return {k: group_eval_metric(v) for k, v in self.group_to_label.items()}

    def print_report(self):
        print('**all metric**')
        print(self.all_metric.print_eval_metric())

        print('\n**non neu metric**')
        print(self.non_neu_metric.print_eval_metric())

        print('\n**group metric**')
        for k, v in self.group_metric.items():
            print('[group ', k, ']')
            v.print_eval_metric()
        print('')