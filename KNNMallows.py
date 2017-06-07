"""
KNN Mallows model
implementing Cheng, W., Huhn, J. and Hullermeier, E., 2009, June. Decision tree and instance-based learning for label ranking [1]
"""
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
from datetime import timedelta
import sys

from KNNPlackettLuce import KNN
from DecisionTreeMallows import MM
from DecisionTreeMallows import rankO2New
from DecisionTreeMallows import rankN2Old
import ReadData
from PerfMeasure import perfMeasure


class KNNMallows(KNN):
    def aggregate(self, neighbors):
        return Mallows().fit(self.y[neighbors,:]).predict()


class Mallows(object):
    """
    input are rankings in old form
    """
    def __init__(self):
        self.Nclass = 0
        self.median = None

    def fit(self, y_s, max_iteration = 100, iter_out = True):
        # data structure and transformation #
        self.Nclass = y_s.shape[1]
        ranks = y_s.tolist()
        ranks = map(rankO2New, ranks)
        _, self.median, iter = MM(ranks, max_iter = max_iteration, iter_out = iter_out, theta_calculate=False)
        print "converge in ", iter
        return self

    def predict(self):
        return np.array(rankN2Old(self.median))


def crossValidate(x, y, cv=5, K=None):
    """
    :param y: N*L ranking vectors
    :return:
    """
    results = {"perf": []}

    ## cross validation ##
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(x):
        x_train = x[train, :]
        y_train = y[train, :]
        x_test = x[test, :]
        y_test = y[test, :]

        y_pred = KNNMallows(K=K).fit(x_train, y_train).predict(x_test)
        # print y_pred ### test
        results["perf"].append(perfMeasure(y_pred, y_test, rankopt=True))
        # print results["perf"][-1]

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":
        K = 20

        dataset = "calhousing"
        # dataset = sys.argv[1]
        x, y = ReadData.readSyntheticData("data/synthetic/" + dataset)

        start = datetime.now()
        result = crossValidate(x[:100,:],y[:100,:],K=K)
        duration = datetime.now() - start

        print duration.total_seconds()
        print result

        with open("results/result_KNNMallows_synthetic.txt", "a") as f:
            f.write("K = %d\n" % K)
            f.write("dataset: synthetic %s\n" % dataset)
            f.write("time = %f\n" % duration.total_seconds())
            f.write(str(result)+"\n")