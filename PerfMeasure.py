"""
calculate performance measures
Three perf measures used correspond to those in perf_list in perfMeaure
acc@3: perf_list["acc3"]
tau: perf_list["kendalltau"]
GMR: perf_list["g_mean_pair"]
"""

import numpy as np
import itertools
from scipy.stats.mstats import gmean
import math
from ReadData import rankOrder

NOISE = 0.001
NONERECALL = 1e-3

def perfMeasure(y_pred, y_test, rankopt = False):
    """
    calculate performance measure for label ranking
    with normal ranking input y_pred and y_test, rankopt = True
    with votes for each label input y_pred and y_test (for LogR and NAIVE specifically), rankopt = False
    """
    # performance measurement #
    # with background noise probability distribution
    if type(y_test)!= np.ndarray:
        y_test = np.array(y_test)
    Nsamp = y_test.shape[0]
    Nclass = y_test.shape[1]

    perf_list={"acc3":0, "dcg":1, "llh":2, "recall":3, "recallsub": 3+Nclass, "Nsc": 3+2*Nclass, "kld": 3 + 3*Nclass,
               "g_mean":4+3*Nclass, "kendalltau": 5+3*Nclass,
               "recallpair": 6+3*Nclass, "Nsp": 6+3*Nclass+Nclass*Nclass, "g_mean_pair": 6+3*Nclass+2*Nclass*Nclass}
    Nperf = max(perf_list.values())+1
    perf=[0 for i in range(Nperf)]

    if not rankopt:
        rank_test = map(rankOrder,y_test)
        rank_pred = map(rankOrder,y_pred)
    else:
        if type(y_test) == np.ndarray:
            rank_test = y_test.tolist()
        else:
            rank_test = y_test
        if type(y_pred) == np.ndarray:
            rank_pred = y_pred.tolist()
        else:
            rank_pred = y_pred

    # acc3 #
    if Nclass>=3:
        for i in range(Nsamp):
            acc3 = 1
            for j in range(3):
                if rank_test[i][j]>=0 and rank_pred[i][j]>=0 and rank_test[i][j]!=rank_pred[i][j]:
                    acc3 = 0
            perf[perf_list["acc3"]] += acc3
        perf[perf_list["acc3"]]=perf[perf_list["acc3"]]/(1.0*Nsamp)
    else:
        print "cannot calculate acc@3 for less than 3 classes"

    # recall #
    recall = map(recallAll, itertools.izip(rank_pred, rank_test))
    recall = np.array(recall)
    recall = np.mean(recall, axis=0)
    for i in range(Nclass):
        perf[perf_list["recall"] + i] = recall[i]

    # recallsub #
    recall_sub, Nsamp_class = recallSub(rank_pred, rank_test)
    for i in range(Nclass):
        perf[perf_list["recallsub"] + i] = recall_sub[i]
        perf[perf_list["Nsc"] + i] = Nsamp_class[i]

    # recallpair #
    recall_pair, Nsamp_pair = recallPair(rank_pred, rank_test)
    cnt = 0
    for i in range(Nclass):
        for j in range(Nclass):
            perf[perf_list["recallpair"] + cnt] =recall_pair[i][j]
            perf[perf_list["Nsp"] + cnt] = Nsamp_pair[i][j]
            cnt += 1

    # G-Mean-pair #
    recall_pair = np.array(recall_pair)
    recall_pair_masked = np.ma.masked_invalid(recall_pair)
    g_mean_pair = gmean(recall_pair_masked, axis=None)
    perf[perf_list["g_mean_pair"]] = g_mean_pair

    # G-Mean #
    recall_sub = np.array(recall_sub)
    recall_sub_masked = np.ma.masked_invalid(recall_sub)
    g_mean = gmean(recall_sub_masked)
    # g_mean = gmean(recall)
    perf[perf_list["g_mean"]] = g_mean

    # Kendall's Tau traditional one#
    perf[perf_list["kendalltau"]] = KendallTau(rank_pred, rank_test)
    # #
    if rankopt:
        return perf

    # --------------- for probability output ------------------------ #
    # llh (log-likelihood)#
    y_pred_noise=addNoise(y_pred)# add noise prior
    # y_pred_noise = y_pred
    for i in range(Nsamp):
        llh = 0
        for j in range(Nclass):
            if y_test[i][j]==0:
                maxllh=0
            else:
                maxllh=y_test[i][j]*math.log(y_test[i][j])
            llh += (y_test[i][j]*math.log(y_pred_noise[i][j])-maxllh)
        try:
            llh = llh + sum(y_test[i])*math.log(sum(y_test[i]))
        except ValueError, e:
            print y_test[i]
            raise e
        perf[perf_list["llh"]] += llh
    perf[perf_list["kld"]] = perf[perf_list["llh"]]/(1.0*np.sum(y_test)) ## normalized by total emoticons
    perf[perf_list["llh"]]=perf[perf_list["llh"]]/(1.0*Nsamp) ## normalized by # samples

    # dcg #
    for i in range(Nsamp):
        dcg = 0
        maxdcg = 0
        for j in range(Nclass):
            if rank_pred[i][j]>=0:
                dcg += y_test[i][rank_pred[i][j]]/math.log(j+2)
            if rank_test[i][j]>=0:
                maxdcg += y_test[i][rank_test[i][j]]/math.log(j+2)
        perf[perf_list["dcg"]] += dcg*1.0/maxdcg
    perf[perf_list["dcg"]]= perf[perf_list["dcg"]] /(1.0*Nsamp)

    return perf


def KendallTau(rank_pred, rank_test):
    # traditionally defined based on concordant and discordant pairs
    # that is missing pairs are excluded
    if type(rank_pred)!=list:
        rank_pred = rank_pred.tolist()
    if type(rank_test)!=list:
        rank_test = rank_test.tolist()
    Nsamp = len(rank_test)
    Nrank = len(rank_test[0]) # the length of each complete rank

    tau = [np.nan for i in range(Nsamp)]
    for samp in range(Nsamp):
        cor = 0
        dis = 0
        rt = rank_test[samp]
        rp = rank_pred[samp]
        for i in range(Nrank):
            emoti_i = rt[i]
            if emoti_i < 0:  # no emoti at the index, emoti_j<0 too
                break   # same for lower ranking emoti
            for j in range(i+1, Nrank):
                emoti_j = rt[j]
                if emoti_j < 0:
                    break # use traditional Kendall, no such constraint exists
                if emoti_i not in rp: # higher ranked emoti not exist, discordant
                    break # use traditional Kendall, no such constraint exists
                if emoti_j not in rp:
                    break # use traditional Kendall, no such constraint exists
                else:
                    pred_i = rp.index(emoti_i)
                    pred_j = rp.index(emoti_j)
                    if pred_i < pred_j:
                        cor += 1
                    else:
                        dis += 1
        if cor + dis >= 1:
            tau[samp] = (cor - dis) * 1.0 / (cor + dis)
    tau = np.array(tau,dtype = "float")
    return np.nanmean(tau)


def recallAll(two_rank):
    # calculate recall for all classes with input (rank1,rank2)
    # consider all posts
    pred,test = two_rank
    if type(pred)!=list:
        pred = pred.tolist()
    if type(test)!=list:
        test = test.tolist()
    Nclass = len(test)
    recall = [1.0 for i in range(Nclass)]
    for rank_test in range(Nclass):
        emoti = test[rank_test]
        if emoti<0:
            continue
        if emoti not in pred:
            recall[emoti] = 0.0
            continue
        rank_pred = pred.index(emoti)
        if rank_pred > rank_test:
            recall[emoti] = 0.0
        ### test
        # print "rank_test", rank_test, "emoti", emoti, "rank_pred", rank_pred
    return recall

def recallSub(rank_pred, rank_test):
    # consider recall of each emoticon in those posts it appears in
    if type(rank_pred)!=list:
        rank_pred = rank_pred.tolist()
    if type(rank_test)!=list:
        rank_test = rank_test.tolist()
    Nclass = len(rank_pred[0])
    recall = [[] for i in range(Nclass)]
    Nsamp_class = [0.0 for i in range(Nclass)]  # #samples each class appears in
    Nsamp = len(rank_pred)

    for i in range(Nsamp):
        for emoti in range(Nclass):
            if emoti not in rank_test[i]:
                continue    # no such emoticon appears
            Nsamp_class[emoti] += 1
            rt = rank_test[i].index(emoti)
            if emoti not in rank_pred[i]:
                recall[emoti].append(0.0)
                continue
            rp = rank_pred[i].index(emoti)
            if rp <= rt:
                recall[emoti].append(1.0)
            else:
                recall[emoti].append(0.0)
    for i in range(Nclass):
        if Nsamp_class[i] < 1.0:
            recall[i] = np.nan
        else:
            recall[i] = sum(recall[i])/Nsamp_class[i]
            if recall[i] < NONERECALL:
                print "NONERECALL:", recall[i], "for emoticon:", i
                recall[i] = NONERECALL

    return recall, Nsamp_class


def recallPair(rank_pred, rank_test):
    # include labels not appearing in rank #
    if type(rank_pred)!=list:
        rank_pred = rank_pred.tolist()
    if type(rank_test)!=list:
        rank_test = rank_test.tolist()
    Nclass = len(rank_pred[0])
    recall = [[0 for i in range(Nclass)] for j in range(Nclass)]
    Nsamp_pair = [[0 for i in range(Nclass)] for j in range(Nclass)]
    Nsamp = len(rank_pred)

    for i in range(Nsamp):
        for emoti_a in range(Nclass):
            for emoti_b in range(emoti_a + 1, Nclass):
                prior = None
                latter = None
                if emoti_a not in rank_test[i]:
                    if emoti_b not in rank_test[i]:
                        continue
                    else:
                        prior = emoti_b
                        latter = emoti_a
                else:
                    if emoti_b not in rank_test[i]:
                        prior = emoti_a
                        latter = emoti_b
                    else:
                        # both appear in rank #
                        ind_a = rank_test[i].index(emoti_a)
                        ind_b = rank_test[i].index(emoti_b)
                        if ind_a < ind_b:
                            prior = emoti_a
                            latter = emoti_b
                        else:
                            prior = emoti_b
                            latter = emoti_a
                if prior is None or latter is None:
                    continue

                Nsamp_pair[prior][latter] += 1

                if prior not in rank_pred[i]:
                    continue
                else:
                    if latter not in rank_pred[i]:
                        recall[prior][latter] += 1
                    else:
                        if rank_pred[i].index(prior) < rank_pred[i].index(latter):
                            recall[prior][latter] += 1
                        else:
                            continue

    for i in range(Nclass):
        for j in range(i+1, Nclass):
            if Nsamp_pair[i][j] < 1:
                recall[i][j] = np.NaN
            else:
                recall[i][j] = float(recall[i][j]+1)/(Nsamp_pair[i][j]+2)
            if Nsamp_pair[j][i] < 1:
                recall[j][i] = np.NaN
            else:
                recall[j][i] = float(recall[j][i]+1)/(Nsamp_pair[j][i]+2)
    for i in range(Nclass):
        recall[i][i] = np.nan
    return recall, Nsamp_pair


def addNoise(dist_list):
    noise = np.array([NOISE for i in range(dist_list.shape[1])])
    return map(lambda x: (x+noise)/sum(x+noise), dist_list)
