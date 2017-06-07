"""
Jason Zhang 02/25/2017 jpz5181@psu.edu
follow Grbovic et al. 2013 IJCAI [1]
"""
import numpy as np
import warnings
from sklearn.model_selection import KFold
import math
import copy
import sys

from PerfMeasure import perfMeasure
import ReadData

class SmpRank(object):
    def __init__(self, K):
        ## intrinsic parameters ##
        self.mfeature = [[] for c in range(K)]
        self.mpairlabel = [[] for c in range(K)]
        self.varfeature = None
        self.varpairlabel = None
        self.K = K
        self.d = None
        self.L = None

        ## intermediate parameters ##
        self.probfeature = [0.0 for c in range(K)]
        self.probpairlabel = [0.0 for c in range(K)]
        self.probsum = 0.0

        ## loss_SMP for previous training set ##
        self.losssmp = np.nan

        ## hyperparameter for SGD ##
        self.decayvarf = np.nan
        self.initvarf = np.nan
        self.decaylearningrate = np.nan
        self.initlearningrate = 0.04 # from [1] empirical #

    def fit(self, x_train, y_train_rank, x_valid = None, y_valid = None, Max_epoch = 100):
        """

        :param x_train: N*d np.array
        :param y_train_rank: N*L np.array ranking vectors
        :return: self
        """
        ## transform ranking to preference matrix ##
        if type(y_train_rank) == np.ndarray:
            y_train_rank = y_train_rank.tolist()
        y_train = map(self.rank2pair, y_train_rank)
        y_train = np.array(y_train, dtype = np.float32)

        ## build validate set if not given ##
        if x_valid is None:
            N0 = x_train.shape[0]
            samps = np.arange(N0)
            np.random.shuffle(samps)
            N_val = N0/10
            x_valid = x_train[samps[:N_val],:]
            y_valid = y_train[samps[:N_val],:,:]
            x_train = x_train[samps[N_val:],:]
            y_train = y_train[samps[N_val:],:,:]

        N = x_train.shape[0]
        d = x_train.shape[1]
        L = y_train.shape[1]
        print "N, d, L: ", N, d, L

        self.initialize(N,d,L, x_train, y_train)

        ## SGD ##
        loss_valid = np.nan
        for epoch in range(Max_epoch):
            samps = np.arange(N)
            np.random.shuffle(samps)
            for t in range(N):
                # complete training set #
                samp = samps[t]
                self.update(x_train[samp], y_train[samp], t = (epoch * N + t))

            # real time result test #
            loss_valid_now = self.losscal(x_valid, y_valid)
            loss_train_now = self.losscal(x_train, y_train)
            print "loss_valid", epoch, loss_valid_now
            print "loss_train", epoch, loss_train_now
            # for SGD correctness test #
            if not np.isnan(self.losssmp) and self.losssmp < loss_train_now:
                warnings.warn("training set loss increase")
                print "last epoch loss: ", self.losssmp
                print "current epoch loss: ", loss_train_now
            elif np.isnan(loss_train_now):
                print "nan train loss"
                print "early stop at epoch", epoch
                return self     # due to decreasing variance
            self.losssmp = loss_train_now
            # stop criterion #
            if not np.isnan(loss_valid) and loss_valid < loss_valid_now:
                print "early stop at the end of epoch: ", epoch
                return self
            elif np.isnan(loss_valid_now):
                print "nan valid loss"
                print "early stop at epoch", epoch
                return self     # due to decreasing variance
            loss_valid = loss_valid_now
        return self

    def predict(self, x_test):
        """
        calculate predicted RANKING for input feature values
        :param x_test:
        :return: y_pred n*L ranking vectors
        """
        ## batch predict ##
        if x_test.ndim != 1:
            N = x_test.shape[0]
            y_rank_pred = [[] for i in range(N)]
            for samp in range(N):
                y_rank_pred[samp] = self.predict(x_test[samp])
            return np.array(y_rank_pred, dtype=np.int8)

        ## weighted average of preference matrices ##
        self.probcal(x_test)
        y_pred = np.zeros([self.L, self.L], dtype = np.float32)
        for k in range(self.K):
            y_pred = y_pred + self.probfeature[k] * self.mpairlabel[k]

        ## aggregate preference to rank ##
        y_rank_pred = self.aggregate(y_pred)

        return y_rank_pred

    def aggregate(self, y):
        """
        transfer preference matrix to ranking
        using TOUGH from [1]
        :param y: L*L np.ndarray
        :return: L ranking
        """
        L = y.shape[0]
        y_rank = []
        labels = [label for label in range(L)]
        y_to_max = copy.deepcopy(y)
        while len(labels)>0:
            # find max value pair in preference matrix #
            ind_max = np.argmax(y_to_max)
            prior, latter = ind_max / L, ind_max % L
            val_max = y_to_max[prior, latter]
            # if val_max == 0:
            #     break
            # for test #
            if prior not in labels:
                print y
                print "y_to_max", y_to_max
                print prior, latter
                print y_rank
                print labels
                raise ValueError("cannot find maximum in remaining labels")
            # search highest agree score position #
            agree_add_max = 0.0
            pos_max = 0
            for pos in range(len(y_rank)+1):
                # rank_temp = [l for l in y_rank]
                # rank_temp.insert(pos, prior)
                agree_add_temp = self.agreeScoreUpdate(y_rank, y, pos, prior)
                # print pos, prior, agree_add_temp ### test ###
                if agree_add_temp >= agree_add_max:
                    agree_add_max = agree_add_temp
                    pos_max = pos
            # update ranking, label set and preference matrix
            y_rank.insert(pos_max, prior)
            # print "y_rank", y_rank### test ###
            del labels[labels.index(prior)]
            y_to_max[prior,:] = -1.0 ## not to be picked again ##
            for ll in y_rank:
                if ll != prior:
                    y[ll, prior] = 0.0
                    y[prior, ll] = 0.0
        # tail abstention #
        for pos in range(len(y_rank), L):
            y_rank.append(-1)
        return y_rank

    def agreeScoreUpdate(self, ranking, preference, pos, prior):
        """
        calculate the agree score change with adding prior to ranking at pos
        :param ranking: current ranking vector
        :param preference: preference matrix L*L np.ndarray
        :return: agree score change
        """
        rank_temp = [label for label in ranking]
        rank_temp.insert(pos, prior)
        # print rank_temp ### test
        agree_add = 0.0
        for i in range(len(rank_temp)):
            if i < pos:
                agree_add += preference[rank_temp[i], prior]
            elif i > pos:
                agree_add += preference[prior, rank_temp[i]]
        return agree_add

    def initialize(self, N, d, L, x_train, y_train):
        ## initialize parameters according to [1] ##
        self.decayvarf = N*5.0
        self.initvarf = float(np.mean(np.linalg.norm(
            x_train - np.mean(x_train, axis = 0),
            axis = 1
        )))
        self.varfeature = self.initvarf
        self.varpairlabel = float(np.mean(np.linalg.norm(
            y_train - np.mean(y_train, axis = 0),
            axis = (1,2)
        )))
        self.decaylearningrate = N*5.0
        self.d = d
        self.L = L

        cf = 0
        cp = 0
        features = []
        pairlabels = []
        for samp in range(N):
            x_samp = x_train[samp]
            y_samp = y_train[samp]
            if not self.check(features, x_samp) and cf < self.K:
                self.mfeature[cf] = x_samp
                features.append(x_samp)
                cf += 1
            if not self.check(pairlabels, y_samp) and cp < self.K:
                self.mpairlabel[cp] = self.complete(y_samp)
                pairlabels.append(y_samp)
                cp += 1
            if cf >= self.K and cp >= self.K:
                break
        if cf < self.K or cp < self.K:
            print "distinct features: ", cf, "distinct pairlabel: ", cp
            raise ValueError("too many prototypes")

    def check(self, list, vector):
        isin = False
        for vect in list:
            if np.array_equal(vect, vector):
                isin = True
        return isin

    def complete(self, y):
        for i in range(self.L):
            for j in range(i+1, self.L):
                if y[i,j] == 0 and y[j, i] == 0:
                    y[i,j] = 0.5
                    y[j,i] = 0.5
        return y

    def update(self, x, y, t):
        ## update decaying parameters ##
        self.varfeature = self.initvarf * (self.decayvarf * 1.0 / (self.decayvarf + t))

        # calculate intermediate parameters #
        self.probcal(x, y)

        ## update ##
        learningrate = self.initlearningrate * \
                       (self.decaylearningrate * 1.0 / (self.decaylearningrate +t))
        for k in range(self.K):
            self.mfeature[k] = self.mfeature[k] - ((learningrate * (self.probsum - self.probpairlabel[k]) *
                                              self.probfeature[k] / (self.probsum * pow(self.varfeature, 2))) *
                                             (x - self.mfeature[k]))
            self.mpairlabel[k] = self.mpairlabel[k] + ((learningrate * (self.probpairlabel[k] * self.probfeature[k]) /
                                                  (self.probsum * pow(self.varpairlabel, 2))) *
                                                 (y - self.mpairlabel[k]))

    def losscal(self, xs, ys):
        """
        calculate losssmp with current parameters value and given xs, ys set
        :param xs: n*d np.ndarray
        :param ys: n*L*L np.ndarray
        :return: losssmp
        """
        losssmp = 0.0
        N = xs.shape[0]
        for samp in range(N):
            self.probcal(xs[samp], ys[samp])
            losssmp += (-np.log(self.probsum)/N)
        return losssmp

    def probcal(self, x, y = None):
        sum_probf = 0.0
        for k in range(self.K):
            self.probfeature[k] = frobeniusGaussianCore(x, self.mfeature[k], self.varfeature)
            sum_probf += self.probfeature[k]
            if y is None:
                continue
            self.probpairlabel[k] = frobeniusGaussianCore(y, self.mpairlabel[k], self.varpairlabel)
        if y is None:
            return
        self.probsum = 0.0
        for k in range(self.K):
            self.probfeature[k] = self.probfeature[k] / sum_probf
            self.probsum += self.probfeature[k] * self.probpairlabel[k]

    def rank2pair(self, ranking, complete=True):
        """
        transform ranking to pairwise comparison matrix
        :param ranking: ranking vector
        :return: np.ndarray
        """
        if type(ranking) == np.ndarray:
            ranking = ranking.tolist()
        L = len(ranking)
        pref = np.zeros([L,L])
        labels = [label for label in range(L)]
        for i in range(L):
            prior = ranking[i]
            if prior < 0:
                break # tail abstention #
            del labels[labels.index(prior)]
            for label in labels:
                pref[prior, label] = 1.0
        # complete preference #
        for i in range(L):
            for j in range(i+1, L):
                if pref[i,j]<0.001 and pref[j,i]<0.001:
                    pref[i,j]=0.5
                    pref[j,i]=0.5
        return pref


def frobeniusGaussianCore(x, m, sigma):
    if sigma == 0.0:
        return 1.0
    return np.exp(-pow(np.linalg.norm(x - m)/sigma, 2))


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

        y_pred = SmpRank(K=K).fit(x_train, y_train).predict(x_test)
        results["perf"].append(perfMeasure(y_pred, y_test, rankopt=True))
        print results["perf"][-1]

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results

def simulateddata(N, L, d):
    x_train = np.random.random([N,d])
    y_train = np.zeros([N,L], dtype=np.int8)
    for i in range(N):
        y_train[i] = np.array(ReadData.rankOrder(x_train[i].tolist()))
    return x_train, y_train


if __name__ == "__main__":
    dataset = "bodyfat"
    # dataset = sys.argv[1]
    x, y = ReadData.readSyntheticData("data/synthetic/" + dataset)
    K = min(max(math.factorial(y.shape[1]-1), math.factorial(y.shape[1])/2), y.shape[0]/50)
    if dataset == "elevators":
        K = 50
    print "dataset, N,d,L,K", dataset, y.shape[0], x.shape[1], y.shape[1], K
    results = crossValidate(x,y,K=K)
    print results
    with open("results/result_SMP_synthetic.txt", "a") as f:
        f.write("dataset: synthetic %s\n" % dataset)
        f.write("K: %d\n" % K)
        f.write(str(results))
        f.write("\n")
