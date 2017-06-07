"""
Implementation based on Dekel, O., Manning, C.D. and Singer, Y., 2003, December. Log-Linear Models for Label Ranking. In NIPS (Vol. 16).    [1]
Using base function proposed in Label Ranking by Learning Pairwise Preferences. 2008, AI.   [2]
"""
import numpy as np
import copy
from sklearn.model_selection import KFold
import sys

from SMPrank import SmpRank
from PerfMeasure import perfMeasure
import ReadData

MAX_ITERATION = 100
THRESHOLD = 0.001

class logLinear(object):
    def __init__(self):
        # Model parameters #
        self.K = 0  # # base functions
        self.lamda = None
        self.Nclass = 0
        self.Dfeature = 0

        # intermediate parameters #
        self.p = None   # base function difference for each pair of each instance
        self.ro = 0     # max of all self.p
        self.wp = None
        self.wm = None
        self.loss = 0
        self.lamda_old = None   # keep track of lamda in previous iteration

    def fit(self, x, y, max_iteration = MAX_ITERATION, threshold = THRESHOLD):
        """
        input y: ranking vectors np.ndarray([Nsamp, Nclass])
        """

        self.inputwrite(x,y)
        y_pref = self.rank2pref(y)
        self.initialize()
        self.loss = self.losscal(x,y_pref)
        loss_old = self.loss
        self.setold()
        for iter in range(max_iteration):
            # print "loss at iter: ", iter, self.loss
            self.update(x, y_pref)
            self.loss = self.losscal(x,y_pref)
            # check convergence #
            if int(loss_old/threshold) <= int(self.loss/threshold):
                print "old loss: ", loss_old
                print "current loss: ", self.loss
                print "end at iter: ", iter
                break
            else:
                loss_old = self.loss
                self.setold()
        return self

    def inputwrite(self, x, y):
        """
        write input info
        """
        self.Nclass = y.shape[1]
        self.K = self.Nclass
        self.Dfeature = x.shape[1]

    def rank2pref(self, y):
        """
        transforming batch of ranking vectors to preference matrix
        """
        smp = SmpRank(K=1)
        rank_list = y.tolist()
        pref_list = map(lambda x: smp.rank2pair(np.array(x)), rank_list)
        return np.array(pref_list)

    def initialize(self):
        self.lamda = np.zeros([self.K, self.Dfeature])
        return self

    def losscal(self, x, y):
        """
        calculate loss value to check convergence
        """
        loss_total = 0.0
        for isamp in range(x.shape[0]):
            p = np.zeros([self.Nclass, self.Nclass, self.K, self.Dfeature]) # default 0 for not existing edges and base function defined in [2]
            for prior in range(self.Nclass):
                for latter in range(prior + 1, self.Nclass):
                    if y[isamp, prior, latter] > 0.99:
                        # == 1.0 #
                        p[prior, latter, prior, :] = - x[isamp, :]
                        p[prior, latter, latter, :] = x[isamp, :]
                    elif y[isamp, latter, prior] > 0.99:
                        # == 1.0 #
                        p[latter, prior, latter, :] = - x[isamp, :]
                        p[latter, prior, prior, :] =  x[isamp, :]
            f = np.tensordot(p, self.lamda, axes = ([-2,-1],[0,1]))
            loss_isamp = np.sum(np.log(1.0 + np.exp(f)))
            loss_total += loss_isamp
        return loss_total

    def setold(self):
        self.lamda_old = copy.deepcopy(self.lamda)
        return self

    def update(self, x, y):
        wp = np.zeros(self.lamda.shape, dtype = np.float64)
        wm = copy.deepcopy(wp)
        for isamp in range(x.shape[0]):
            # calculate p parameters for each sample #
            p = np.zeros([self.Nclass, self.Nclass, self.K, self.Dfeature]) # default 0 for not existing edges and base function defined in [2]
            for prior in range(self.Nclass):
                for latter in range(prior + 1, self.Nclass):
                    if y[isamp, prior, latter] > 0.99:
                        # == 1.0 #
                        p[prior, latter, prior, :] = - x[isamp, :]
                        p[prior, latter, latter, :] = x[isamp, :]
                    elif y[isamp, latter, prior] > 0.99:
                        # == 1.0 #
                        p[latter, prior, latter, :] = - x[isamp, :]
                        p[latter, prior, prior, :] = x[isamp, :]
            # update self.ro #
            ro = np.amax(np.sum(np.sum(np.abs(p), axis=-1), axis=-1))
            if ro > self.ro:
                self.ro = ro
            # calculate q for each sample #
            q_temp = np.exp(np.tensordot(self.lamda, p, axes = ([0, 1], [-2, -1])))
            q = np.divide(q_temp, 1.0 + q_temp)
            # update wp, wm for each sample #
            pp = copy.deepcopy(p)
            pp[pp < 0] = 0
            pm = copy.deepcopy(p)
            pm[pm > 0] = 0
            wp += np.tensordot(q,pp, axes = ([0,1],[0,1]))
            wm += np.tensordot(q,- pm, axes = ([0,1],[0,1]))
        self.lamda = self.lamda - 0.5/self.ro*np.log(np.divide(wp, wm))
        return self

    def predict(self, x_test):
        f = np.tensordot(x_test, self.lamda, axes=(1,1))
        return ReadData.label2Rank(f.tolist())


def crossValidate(x, y, cv=5):
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

        y_pred = logLinear().fit(x_train, y_train).predict(x_test)
        # print "y_pred", y_pred
        # print "y_test", y_test
        results["perf"].append(perfMeasure(y_pred, y_test, rankopt=True))
        print results["perf"][-1]

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results

if __name__ == "__main__":
    dataset = "Random_Normal_User.txt"
    x,y = ReadData.dataFacebook("data/" + dataset )
    y = ReadData.label2Rank(y)
    results = crossValidate(x[:1000,:],y[:1000,:])
    print results
    with open("results/result_logLinear.txt", "a") as f:
        f.write("dataset: %s\n" % dataset)
        f.write(str(results))
        f.write("\n")
