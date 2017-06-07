"""
Implementing Hullermeier, E., Furnkranz, J., Cheng, W. and Brinker, K., 2008. Label ranking by learning pairwise preferences. Artificial Intelligence, 172(16), pp.1897-1916.
"""


import numpy as np
from sklearn.model_selection import KFold
import sys

import logRegression as LogR
import ReadData
from PerfMeasure import perfMeasure


class RankPairPref(object):
    def __init__(self):
        self.x_train = None # training features
        self.y_train_pair = None # training pairwise comaprisons
        # abstention parameters #
        self.Abstention = False
        self.k = 2

    def fit(self, x_train, y_train, k = 2, Abstention = False):
        self.Abstention = Abstention
        self.k = k
        y_train_score = np.array(map(self.rank2score, y_train.tolist()))
        self.x_train, self.y_train_pair = self.score2pair(x_train, y_train_score, k = self.k, Abstention = self.Abstention)
        return self

    def predict(self, x_test):
        pair_pref_prob = self.pairpref(self.x_train, self.y_train_pair, x_test)
        ppp = pair_pref_prob.tolist()
        ranks = np.array(map(self.pair2rank, ppp))
        return ranks

    def rank2score(self, rank):
        """
        transforming rank vector to score with (d-i),
        where d is the highest rank +1, and i is the rank position
        only complete rank!!!
        specific for transforming ranking vector data into rpp framework
        :param rank: rank vector with label id on its ranking position
        :return: score label vector
        """
        Nclass = len(rank)
        scores = [0 for i in range(Nclass)]
        labels = [True for j in range(Nclass)]
        for pos in range(Nclass):
            if rank[pos] >= 0:
                scores[rank[pos]] = Nclass - pos
                labels[rank[pos]] = False
        for label in range(Nclass):
            if labels[label]:
                scores[label] = -1 # keep labels in tail abstention lowest score
        return scores

    def score2pair(self, x, y, k=2, Abstention=False):
        """
        k is the enhance factor for preference pairs over no-prefer pairs
        the bigger k is, the smaller the effect of no-prefer pairs,
        kind of inverse of laplace smooth

        y is the score data
        """
        # consider not appearing emoticons as ranked lowest #
        Nsamp = y.shape[0]
        Nclass = y.shape[1]
        Nfeature = x.shape[1]
        # no-prefer pairs are included #
        y_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
        x_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
        # for case no-prefer pairs are excluded#
        x_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
        y_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
        # i,j value = 1 if emoticon i ranks higher than j, 0 otherwise #
        for samp in range(Nsamp):
            for i in range(Nclass):
                for j in range(i + 1, Nclass):
                    if y[samp, i] > y[samp, j]:
                        for _ in range(k):
                            y_enlarge[i][j].append(1)
                            x_enlarge[i][j].append(x[samp])
                        y_list[i][j].append(1)
                        x_list[i][j].append(x[samp])
                    elif y[samp, i] < y[samp, j]:
                        for _ in range(k):
                            y_enlarge[i][j].append(0)
                            x_enlarge[i][j].append(x[samp])
                        y_list[i][j].append(0)
                        x_list[i][j].append(x[samp])
                    else:  # y[samp,i] == y[samp,j]
                        for value in [0, 1]:
                            y_enlarge[i][j].append(value)
                            x_enlarge[i][j].append(x[samp])

        if Abstention:
            # include no-preference pair #
            return x_enlarge, y_enlarge
        else:
            # exclude no-preference pair #
            return x_list, y_list

    def pairpref(self, x_train, y_train, x_test):
        """
        x_train, y_train in the form of output of score2pair
        x_test should be ordinary Nsamp*Nfeature
        """
        Nclass = len(y_train)
        Nsamp_test = len(x_test)
        pair_pref_prob = np.zeros(shape=[Nsamp_test, Nclass, Nclass])
        for i in range(Nclass):
            for j in range(i + 1, Nclass):
                ## handle extreme cases ##
                if len(y_train[i][j]) == 0:
                    pair_pref_prob[:, i, j] = 0.5 * np.ones(Nsamp_test, dtype="float")
                    continue
                if 0 not in y_train[i][j]:
                    pair_pref_prob[:, i, j] = np.ones(Nsamp_test, dtype="float")
                    continue
                if 1 not in y_train[i][j]:
                    pair_pref_prob[:, i, j] = np.zeros(Nsamp_test, dtype="float")
                    continue
                prob, coef, intercept = LogR.logRegFeatureEmotion(x_train[i][j], y_train[i][j], x_test)
                pair_pref_prob[:, i, j] = prob[:, 1]
        # fill pair_pref_prob matrix #
        for i in range(Nclass):
            for j in range(0, i):
                pair_pref_prob[:, i, j] = 1 - pair_pref_prob[:, j, i]
        return pair_pref_prob

    def pair2rank(self, pair_pref_prob_samp):
        # list(Nclass * Nclass)#
        # borda count #
        ppps = pair_pref_prob_samp
        Nclass = len(ppps)
        for i in range(Nclass):
            if ppps[i][i] != 0:
                print "warning: self comparison", i, ppps[i][i]
                ppps[i][i] = 0
        rscore = map(sum, ppps)
        # print "rscore", rscore
        rank = ReadData.rankOrder(rscore)
        return rank


def crossValidate(x,y,cv=5, k=2, Abstention = False):
    results = {"perf":[]}
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        y_pred = RankPairPref().fit(x_train,y_train, k = k, Abstention = Abstention).predict(x_test)
        results["perf"].append(perfMeasure(y_pred, y_test, rankopt=True))
    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        results[key] = [mean, std]
    return results


if __name__ == "__main__":

    dataset = "bodyfat"
    # dataset = sys.argv[1]
    x, y = ReadData.readSyntheticData("data/synthetic/" + dataset)
    result = crossValidate(x, y)

    # file = open("results/result_LWR_synthetic.txt", "a")
    # file.write("dataset: synthetic %s\n" % dataset)
    # file.write("number of samples: %d\n" % x.shape[0])
    # file.write("CV: %d\n" % 5)
    # file.write(str(result) + "\n")
    # file.close()
    print result