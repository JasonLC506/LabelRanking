"""
implementing label wise ranking algo in Cheng, W., Henzgen, S. and Hullermeier, E., 2013. Labelwise versus Pairwise Decomposition in Label Ranking. In LWA (pp. 129-136).
"""

from sklearn.model_selection import KFold
import numpy as np
import sys

from munkres import Munkres
import ReadData
from PerfMeasure import perfMeasure
import logRegression as LogR


class LabelWiseRanking(object):
    def __init__(self):
        self.Nclass = 0
        self.x_train = None # training features
        self.y_train_pos = None # training targets
        self.valid_samp = None # info of training targets
        self.miss = None    # info of training targets

    def fit(self, x_train, y_train):
        self.Nclass = y_train.shape[1]
        self.x_train = x_train
        self.y_train_pos, self.valid_samp, self.miss = self.rank2position(y_train)
        return self

    def predict(self, x_test):
        Nsamp_test = x_test.shape[0]
        posproblist_labels = [[] for label in range(self.Nclass)]

        ## pos prob predict for each label ##
        for label in range(self.Nclass):
            # get label position distribution estimate by logistic regression #
            posprob, _a, _b = LogR.logRegFeatureEmotion(self.x_train[self.valid_samp[:, label], :],
                                                        self.y_train_pos[self.valid_samp[:, label], label], x_test)
            # complete missing position #
            posprob_complete = self.complete(posprob, self.miss[label])
            posproblist_labels[label] = list(posprob_complete)

        posproblist_labels = np.array(posproblist_labels, dtype=np.float32)

        ## aggregation ##
        rankings = [[] for samp in range(Nsamp_test)]
        for samp in range(Nsamp_test):
            labelposprob = posproblist_labels[:, samp, :]
            # calculate loss matrix #
            labelposloss = self.posprobloss(labelposprob)
            # find ranking minimize loss #
            rankings[samp] = self.aggregate(labelposloss)
        # print rankings### test
        return rankings

    def posprobloss(self, labelposprob):
        """
        here spearman footrule loss for ranking is used
        :param labelposprob: np.ndarray
        :return: labelposloss: np.ndarray([Nclass, Nclass]), expected spearman footrule loss for each label and each position based on labelposprob
        """
        Nclass = labelposprob.shape[0]
        labelposloss = [[-1 for pos in range(Nclass)] for label in range(Nclass)]
        for label in range(Nclass):
            for pos in range(Nclass):
                loss = 0.0
                for prob_pos in range(Nclass):
                    loss += (abs(prob_pos - pos) * labelposprob[label, prob_pos])
                labelposloss[label][pos] = loss
        return np.array(labelposloss, dtype=np.float32)

    def aggregate(self, labelposloss):
        ## using Hungarian algorithm ##
        # print labelposloss ### test
        matrix = labelposloss.tolist()
        Nclass = len(matrix)
        m = Munkres()
        indexs = m.compute(matrix)

        ## transform to rank ##
        ranking = [-1 for i in range(Nclass)]
        for label, pos in indexs:
            ranking[pos] = label
        return ranking

    def rank2position(self, y_train):
        """
        :param y_train: np.array(np.int32) n*d with ranking vector each position as ranking position
        :return: y_train_pos: np.ndarray([Nsamp, Nclass]) for each samp, and each label, what is the ranking position, if missing, -1;
                 valid_samp:  np.ndarray([Nsamp, Nclass]) for each samp, and each label, whether it appears;
                 miss:        np.ndarray([Nclass, Nclass]) for each label and position pair, whether it exists at least in one instance in y_train
        """
        y_train.tolist()
        Nsamp = len(y_train)
        Nclass = len(y_train[0])
        y_train_pos = [[-1 for label in range(Nclass)] for samp in range(Nsamp)]
        valid_samp = [[False for label in range(Nclass)] for samp in range(Nsamp)]
        for samp in range(Nsamp):
            ranking = y_train[samp]
            for pos in range(Nclass):
                if ranking[pos] < 0:
                    break
                else:
                    label = ranking[pos]
                    y_train_pos[samp][label] = pos
                    valid_samp[samp][label] = True
        y_train_pos = np.asarray(y_train_pos, dtype=np.int32)
        valid_samp = np.asarray(valid_samp, dtype=bool)

        miss = [[True for p in range(Nclass)] for label in range(Nclass)]
        for samp in range(Nsamp):
            for label in range(Nclass):
                if valid_samp[samp,label]:
                    miss[label][y_train_pos[samp,label]] = False

        return y_train_pos, valid_samp, miss

    def complete(self, posprob, miss):
        """
        for missing label-position pairs, add position possibility 0.0 to corresponding label
        """
        Nsamp = posprob.shape[0]
        a = posprob.tolist()
        # print type(a), a
        # print "miss", miss
        Nclass = len(miss)
        for i in range(Nclass):
            if miss[i]:
                if i < len(a[0]):
                    for samp in range(Nsamp):
                        a[samp].insert(i,0.0)
                else:
                    for samp in range(Nsamp):
                        a[samp].append(0.0)
        # print a
        return np.array(a)


def crossValidate(x,y,cv=5):
    results = {"perf":[]}
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        y_pred = LabelWiseRanking().fit(x_train,y_train).predict(x_test)
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

    file = open("results/result_LWR_synthetic.txt", "a")
    file.write("dataset: synthetic %s\n" % dataset)
    file.write("number of samples: %d\n" % x.shape[0])
    file.write("CV: %d\n" % 5)
    file.write(str(result) + "\n")
    file.close()
    print result