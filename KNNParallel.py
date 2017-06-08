# from KNNPlackettLuce import KNN
from KNNMallows import KNNMallows as KNN # using Mallows model
import threading
import time
import numpy as np
import logRegression as LogR
from PerfMeasure import perfMeasure
import ReadData
from sklearn.model_selection import KFold
import math
from datetime import datetime

THREADS = 20

class myThread (threading.Thread):
    def __init__(self, threadID, x_test, y_pred, KNNobject):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.x_test = x_test
        self.y_pred = y_pred
        self.KNNobject = KNNobject
    def run(self):
        print "Starting " , self.threadID
        singlethreadPredict(self.x_test, self.y_pred, self.KNNobject)
        print "Exiting " , self.threadID

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

        # y_pred = KNN(K=K).fit(x_train, y_train).predict(x_test)
        y_pred = multithreadPredict(x_test, KNN(K=K).fit(x_train, y_train))
        print y_pred

        # print y_pred ### test
        results["perf"].append(perfMeasure(y_pred, y_test, rankopt=True))
        # print results["perf"][-1]

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results

def multithreadPredict(x_test, KNNobject):

    y_preds = [[] for i in range(THREADS)]
    Nsamp = x_test.shape[0]
    Nsamp_thread = math.ceil(Nsamp*1.0/THREADS)

    threads = []
    for i in range(THREADS):
        threads.append(myThread(i, x_test[i*Nsamp_thread:(i+1)*Nsamp_thread], y_preds[i], KNNobject))
    for i in range(THREADS):
        threads[i].start()
    for i in range(THREADS):
        threads[i].join()
    # join results together #
    y_pred = []
    for i in range(THREADS):
        y_pred = y_pred + y_preds[i]
    return np.array(y_pred)

def singlethreadPredict(x_test, y_pred, KNNobject):
    results = KNNobject.predict(x_test)
    for result in results.tolist():
        y_pred.append(result)

if __name__ == "__main__":
    datafile = "data/posts_Feature_Emotion.txt"
    Ks = [10, 40, 80]
    for K in Ks:
        print K, "start at ", datetime.now()
        x,y = ReadData.dataFacebook(datafile)
        y = np.array(map(ReadData.rankOrder, y.tolist()))
        # x, y = readSushiData()
        # x,y = x[:1000, :], y[:1000, :]

        start = datetime.now()
        result = crossValidate(x,y,K=K)
        duration = datetime.now() - start

        print duration.total_seconds()
        print result

        with open("results/result_KNNMallows.txt", "a") as f:
            f.write("K = %d\n" % K)
            f.write("data = %s\n" % datafile)
            f.write("time = %f\n" % duration.total_seconds())
            f.write(str(result)+"\n")
