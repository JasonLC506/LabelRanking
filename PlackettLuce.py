"""
Implementing the algorithm in
Cheng, W., Hullermeier, E. and Dembczynski, K.J., 2010. Label ranking methods based on the Plackett-Luce model. ICDM [1]
"""

import numpy as np
from ReadData import rankOrder

class PlackettLuce(object):
    def __init__(self):
        self.Nclass = 0
        self.v = None

        ## for fitting ##
        self.llh = None
        self.v_old = None
        self.threshold = 0.01

    def fit(self, y_s, threshold = 0.01, max_iteration = 100):
        self.threshold = threshold
        self.Nclass = y_s.shape[1]

        # initialize parameters #
        self.initialize(y_s)
        self.setold()

        # calculate initial llh #
        self.llh = self.loglikelihood(y_s)
        llh_old = self.llh
        
        for iteration in range(max_iteration):
            # self.printmodel()
            # check convergent #
            if iteration > 0:
                if self.llh < llh_old + threshold:
                    if self.llh < llh_old:
                        print self.llh
                        print llh_old
                        raise ValueError("wrong iteration")
                    else:
                        print "converge in ", iteration
                        break
                else:
                    llh_old = self.llh
            
            Q_before = self.Qcalculate(y_s) ### test for optimization
            self.update(y_s)
            Q_after = self.Qcalculate(y_s) ### test
            try:
                assert Q_after >= Q_before     ### test
            except AssertionError, e:
                print "before: ", Q_before
                print "after: ", Q_after

            self.setold()
            self.llh = self.loglikelihood(y_s)
        return self

    def predict(self):
        return rankOrder(self.v.tolist())

    def probability(self, y):
        y_s = y.reshape([1,y.shape[0]])
        llh = self.loglikelihood(y_s)
        prob = np.exp(llh)
        return prob

    def initialize(self, y_s):
        self.v = np.random.random(self.Nclass)
        return self

    def setold(self):
        self.v_old = self.v
        return self

    def loglikelihood(self, y_s):
        llh = 0.0
        for i in range(y_s.shape[0]):
            llh_samp = 0.0
            for m in range(y_s.shape[1]):
                if y_s[i,m] >= 0:
                    llh_samp += np.log(self.v[y_s[i,m]])
                else:
                    break
                part_sum = 0.0
                for m_ in range(m, y_s.shape[1]):
                    if y_s[i,m_] >= 0:
                        part_sum += self.v[y_s[i,m_]]
                    else:
                        break
                llh_samp = llh_samp - np.log(part_sum)
            llh += llh_samp
        return llh
    
    def Qcalculate(self, y_s):
        Q = 0.0
        for i in range(y_s.shape[0]):
            Q_samp = 0.0
            for m in range(y_s.shape[1]):
                if y_s[i,m] >= 0:
                    Q_samp += np.log(self.v[y_s[i,m]])
                else:
                    break
                part_sum = 0.0
                part_sum_old = 0.0
                for m_ in range(m, y_s.shape[1]):
                    if y_s[i,m_] >= 0:
                        part_sum += self.v[y_s[i,m_]]
                        part_sum_old += self.v_old[y_s[i,m_]]
                    else:
                        break
                Q_samp = Q_samp - part_sum/part_sum_old
            Q += Q_samp
        return Q
    
    def printmodel(self):
        print "v"
        print self.v
        print "llh"
        print self.llh
        
    def update(self, y_s):
        A = np.zeros(self.Nclass, dtype=np.float64)
        B = np.zeros(self.Nclass, dtype=np.float64)
        for samp in range(y_s.shape[0]):
            y = y_s[samp].tolist()
            for label in range(self.Nclass):
                if label not in y:
                    continue
                # label is in y #
                A[label] += 1
                y_index = y.index(label)
                for m in range(y_index+1):
                    sum_old = 0.0
                    for m_ in range(m, self.Nclass):
                        if y[m_] >= 0:
                            sum_old += self.v_old[y[m_]]
                        else:
                            break
                    B[label] += 1.0/sum_old
        for label in range(self.Nclass):
            if B[label]==0.0:
                if A[label]==0.0:
                    self.v[label]=0.0
                else:
                    raise ValueError("B and A not consistent")
            else:
                self.v[label]=A[label]/B[label]
        return self

def synthetic(Nsamp, v_m):
    v_m_m = np.mean(v_m)
    y_s = []
    for i in range(Nsamp):
        v = np.random.random(v_m.shape[0])*v_m_m*3 + v_m
        PL = PlackettLuce()
        PL.v = v
        rnd = np.random.random()
        y = PL.predict()
        for pos in range(v_m.shape[0]):
            if pos >= rnd*v_m.shape[0]:
                y[pos] = -1
        y_s.append(y)
    return np.array(y_s)

if __name__ == "__main__":
    y_s = synthetic(10, np.array([3,2,1]))
    print y_s
    print PlackettLuce().fit(y_s).predict()