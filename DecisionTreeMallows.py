from datetime import datetime
from datetime import timedelta
from scipy.stats import rankdata
from scipy.optimize import ridder
import random
import math
import numpy as np
from sklearn.model_selection import KFold

import DT as DTme
from DT import orderList
from DT import alphaList
from PerfMeasure import perfMeasure
import ReadData

"""
rankform = [[highest_possible_rank, lowest_possible_rank] for emoticon in emoticon_list]
"""

######################### Mallows model fit ##################################
# also used in KNNMallows #

def MM(ranks, max_iter = 10, iter_out = False, theta_calculate = True):
    """
    The modified MM algorithm proposed for incomplete ranks
    :param ranks: y in ranks in given nodes
    :return: the deviation theta and the node result (median)
    """
    if type(ranks)==np.ndarray:
        ranks = ranks.tolist()
    # start = datetime.now() ### test
    median = MMInit(ranks)
    # duration = datetime.now()-start ### test
    # print "init_time: ", duration.total_seconds()###test
    flag_cvg = False
    for iter in range(max_iter):
        ranks_cplt = []
        # start = datetime.now()
        for rank in ranks:
            ranks_cplt.append(MMExt(rank, median))
        # duration = datetime.now()-start
        # print "extension_time: ", duration.total_seconds()
        # start = datetime.now()
        median_new = MMBC(ranks_cplt)
        # duration = datetime.now()-start
        # print "borda count time: ", duration.total_seconds()
        # start = datetime.now()
        if median_new == median:
            # duration = datetime.now() - start
            # print "compare time: ", duration.total_seconds()
            flag_cvg = True
            break
        else:
            median = median_new

    if not flag_cvg:
        print "warning: MM fails to converge"

    # start = datetime.now()
    if theta_calculate:
        theta = MMMallowTheta(ranks_cplt, median)
    else:
        theta = None
    # duration = datetime.now()-start
    # print "find theta time: ", duration.total_seconds()
    if iter_out:
        return theta, median, iter
    return theta, median


def MMInit(ranks):
    # using set of incomplete ranks finding initial median rank for MM algorithm #
    # modified gneralized Borda Count for ranks with abstention in the end #
    # the abstention in the middle is rare and eliminated according label id order #
    Nclass = len(ranks[0])
    init_rank = []
    rscore = [0.0 for i in range(Nclass)]
    for rank in ranks:
        for label in range(Nclass):
            score_label = Nclass - float(rank[label][0] + rank[label][1])/2.0 # key formula #
            rscore[label] += score_label
    init_rank = score2rank(rscore, cplt=True)
    return init_rank

def MMExt(rank,median):
    # Given median rank, find most probable consistent extensions for input incomplete rank #
    # robust for complete rank input #
    Nclass = len(rank)
    ext_rank = [[0,0] for i in range(Nclass)]
    prior = [0 for i in range(Nclass)]
    for label in range(Nclass):
        ind = median[label][0]
        prior[ind] = label
    position_taken = []
    for label in prior:
        for position in range(rank[label][0], rank[label][1] + 1):
            if position not in position_taken:
                position_taken.append(position)
                ext_rank[label][0] = position
                ext_rank[label][1] = position
                break
    return ext_rank


# def MMExt(rank, median):
#     # Given median rank, find most probable consistent extensions for input incomplete rank #
#     # robust for complete rank input #
#     Nclass = len(rank)
#     ext_rank = [rank[i] for i in range(Nclass)]
#     for r in range(Nclass):
#         # for each rank position #
#         compete = []
#         for label in range(Nclass):
#             if ext_rank[label][0] == r and ext_rank[label][1] == r:
#                 break # rank position r is complete
#             elif ext_rank[label][0] == r: # abstention
#                 compete.append(label)
#         if len(compete) == 1:
#             ext_rank[compete[0]][1] = ext_rank[compete[0]][0] # lowest = highest
#         elif len(compete) > 1:
#             highest_label = compete[0]
#             for label in compete:
#                 if median[label][0] < median[highest_label][0]: # rank higher than highest label
#                     highest_label = label
#             for label in compete:
#                 if label == highest_label:
#                     ext_rank[label][1] = ext_rank[label][0]
#                 else:
#                     ext_rank[label][0] += 1
#     return ext_rank


def MMBC(ranks_cplt):
    # using set of complete ranks finding median rank using Borda Count#
    # using the easiest non-weighted Borda Count #
    return MMInit(ranks_cplt)


def MMMallowTheta(ranks_cplt, median):
    """
    :return: the MLE theta parameter in Mallows distribution
    """
    Nsamp = len(ranks_cplt)
    Nclass = len(ranks_cplt[0])
    distances = [discordant(ranks_cplt[i], median) for i in range(Nsamp)]
    distances = np.array(distances, dtype=np.float64)
    dev = np.mean(distances)
    try:
        theta = ridder(MallowsThetaDev, 1e-5, 1e+5, args=(dev, Nclass))
        return theta
    except ValueError, e:
        print "f(a)", MallowsThetaDev(1e-5, dev, Nclass)
        print "f(b)", MallowsThetaDev(1e-5, dev, Nclass)
        print "!!!Not well chosen median"
        raise e



def MallowsThetaDev(theta, dev, Nclass):
    thetadev = Nclass*math.exp(-theta)/(1-math.exp(-theta))-\
               sum([j*math.exp(-j*theta)/(1-math.exp(-j*theta)) for j in range(1,Nclass+1)])
    return (thetadev - dev)



def discordant(rank, rank_ref):
    # the number of discordant pairs #
    dis = 0
    Nclass = len(rank)
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            if (rank[i][0] - rank[j][0])*(rank_ref[i][0]-rank_ref[j][0]) < 0:
                dis += 1
    return dis


def score2rank(rscore, cplt=False):
    """
    :param rscore: the vector of score with each dimension for each label
    :param cplt: True for only complete rank output, False for general output
    :return: rank in rankform
    """
    Nclass = len(rscore)
    rscore_minus = map(lambda x: -x, rscore)
    rank_min = rankdata(rscore_minus, method = "min")-1
    rank_max = rankdata(rscore_minus, method = "max")-1
    rank = [[rank_min[i],rank_max[i]] for i in range(Nclass)]
    if cplt:
        random_prior = [i for i in range(Nclass)]
        random.shuffle(random_prior)
        position_taken = []
        for label in random_prior:
            for position in range(rank[label][0], rank[label][1]+1):
                if position not in position_taken:
                    position_taken.append(position)
                    rank[label][0] = position
                    rank[label][1] = position
                    break
    return rank

######################################################################################

############################## Decision Tree #########################################

def bestSplit(x, y, samples, feature, min_node=1):
    """

    :param x: features
    :param y: np.ndarray of ranks in rankform or rank_old form
    :param samples: np.array
    :param feature: current feature considered
    :param min_node: minimum number of samples in a node
    :return:
    """

    min_var = None
    best_split = 0
    best_sets = []
    best_sets_result = []

    Nsamp = len(samples)

    temp = [(x[samples[i],feature],samples[i]) for i in range(Nsamp)]
    dtype = [("value", float),("index", int)]
    x_ord = np.sort(np.array(temp,dtype=dtype), order = "value")

    old_value = None
    left_size =0
    right_size = Nsamp
    left_samps = np.array([],dtype=np.int16)
    right_samps = np.array([x_ord[s][1] for s in range(Nsamp)],dtype=np.int16)
    for i in range(Nsamp): # avoid 0 split
        value = x_ord[i][0]
        if old_value is not None:
            # print "i = ", i ###test
            if value != old_value:
                # a valid split #
                # print left_samps, right_samps ### test
                # start = datetime.now()### test
                left_result = MM(y[left_samps])
                right_result = MM(y[right_samps])
                # duration = datetime.now() - start ###test
                # print "samps: ", left_size, right_size
                # print "time: ", duration.total_seconds()
                # print "left_result: ", left_result, "right_result: ", right_result, "left_size: ", left_size, "right_size: ", right_size
                variance = 1.0*Nsamp/(left_size*left_result[0]+right_size*right_result[0]) # 1/theta
                if min_var is None or min_var > variance:
                    min_var = variance
                    best_sets = [left_samps, right_samps]
                    best_split = [feature, value] # >= split
                    best_sets_result = [left_result, right_result]

        left_samps = np.append(left_samps, x_ord[i][1])
        right_samps = np.delete(right_samps, 0)
        left_size += 1
        right_size += -1
        old_value = value

    return min_var, best_split, best_sets, best_sets_result


def buildtree(x,y, samples, min_node=1, result_cur = None):
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples) != np.ndarray:
        samples = np.array(samples)
    if len(samples) == 0:
        return DTme.decisionnode()
    ## transform old rank to new rank form
    if y.ndim == 2:
        # rank_old form #
        y = y.tolist()
        temp = map(rankO2New, y)
        y = np.array(temp)


    if result_cur is None:
        result_cur = MM(y[samples])

    if len(samples)<= min_node:
        return DTme.decisionnode(result=result_cur[1])
    # find best split
    best_gain = 0.0
    best_split = []
    best_sets = []
    best_sets_result = []

    N_feature = x.shape[1]
    start = datetime.now() ### test
    for feature in range(N_feature):
        # nlogn selection
        min_var, split, sets, sets_result = bestSplit(x,y,samples,feature)
        if min_var is None:
            continue
        gain = result_cur[0] - min_var
        # print "feature: ", feature, "gain: ", gain, "result_cur: ", result_cur, "min_var: ", min_var ### test
        if gain > best_gain and len(sets[0]) * len(sets[1]) > 0:
            best_gain = gain
            best_split = split
            best_sets = sets
            best_sets_result = sets_result
    duration = datetime.now() - start ### test
    print "Nsamps: ", len(samples)
    print "duration: ", duration.total_seconds()

    if best_gain > 0:
        tb = buildtree(x,y, best_sets[0], min_node = min_node, result_cur = best_sets_result[0])
        fb = buildtree(x,y, best_sets[1], min_node = min_node, result_cur = best_sets_result[1])
        return DTme.decisionnode(feature = best_split[0], value = best_split[1], result = result_cur[1],
                            tb = tb, fb = fb,
                            gain = (tb.gain+fb.gain+best_gain), size_subtree = (tb.size+fb.size))
    else:
        return DTme.decisionnode(result = result_cur[1])


def predict(observation, tree, alpha):
    if tree.tb == None:
        # print "hello world" ### test
        return rankN2Old(tree.result)
    if tree.alpha >= 0:
        if tree.alpha < alpha:
            return rankN2Old(tree.result)
    value = observation[tree.feature]
    if isinstance(value,int) or isinstance(value,float):
        if value>=tree.value:
            branch = tree.tb
        else:
            branch = tree.fb
    else:
        raise("nominal feature not supported")
    return predict(observation,branch,alpha)


def rankN2Old(rank_new):
    # the new rank must be complete #
    Nclass = len(rank_new)
    rank_old = [0 for i in range(Nclass)]
    for label in range(Nclass):
        rank_old[rank_new[label][0]] = label
    return rank_old


def rankO2New(rank_old):
    Nclass = len(rank_old)
    labels = [i for i in range(Nclass)]
    rank_new = [[-1,-1] for i in range(Nclass)]
    non = -1
    for i in range(Nclass):
        label = rank_old[i]
        if label >= 0:
            rank_new[label][0] = i
            rank_new[label][1] = i
            labels.remove(label)
        else:
            non = i
            break
    if non >= 0:
        for label in labels:
            rank_new[label][0] = non
            rank_new[label][1] = Nclass - 1
    return rank_new


# DTme.bestSplit = bestSplit
# DTme.buildtree = buildtree
# DTme.predict = predict


def decisionTree(x_train, y_train, x_test, alpha = None, min_node = 1):
    # x: Nsamp*Nfeature, y: Nsamp*Nrank ranking data
    # prune_list = buildPruneList(x_train, y_train)
    # alpha_list = prune_list.keys()
    # alpha_list.sort()
    # print "start building tree: ", datetime.now() ### test
    tree = buildtree(x_train, y_train, [i for i in range(x_train.shape[0])], min_node=min_node)
    # print "end building tree: ", datetime.now() ### test
    alpha_list = orderList()
    alphaList(tree,alpha_list)
    alpha_max = max(alpha_list.list) + 1
    alpha_list.insert(alpha_max)
    y_pred_list = []
    if alpha == None:
        for alpha_try in alpha_list.list:
            y_pred = []
            for obs in x_test:
                y_pred.append(predict(obs,tree,alpha_try))
            y_pred_list.append(y_pred)
        return alpha_list.list, y_pred_list
    else:
        y_pred=[]
        for obs in x_test:
            y_pred.append(predict(obs, tree, alpha))
        return alpha, y_pred


def crossValidate(x,y, method = "dT",cv=5, alpha = None, min_node = 1):
    #  error measure
    results = []
    if method == "logReg":
        results = {"perf":[], "coef":[], "interc":[]}
    elif method == "dT":
        results = {"alpha": [], "perf":[]}

    # cross validation #
    np.random.seed(1100)
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
    for train,test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        # training and predict

        if alpha == None:
            ## nested select validate and test ##
            # print "start searching alpha:", datetime.now() ### test
            alpha_sel, perf = DTme.hyperParometer(x_train,y_train)
            # print "finish searching alpha:", datetime.now(), alpha ### test
        else:
            alpha_sel = alpha
        result = decisionTree(x_train, y_train, x_test, alpha = alpha_sel, min_node = min_node)

        # performance measure

        alpha_sel, y_pred = result
        results["perf"].append(perfMeasure(y_pred,y_test,rankopt=True))
        results["alpha"].append(alpha_sel)
        print alpha_sel, "alpha"

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":


    x,y = ReadData.dataFacebook("data/nytimes.txt")
    y = ReadData.label2Rank(y)
    start = datetime.now()
    result = crossValidate(x[:100,:],y[:100,:],"dT",cv=5, alpha=0.0, min_node = 1)
    duration = datetime.now() - start
    print "total time: ", duration.total_seconds()
    file = open("results/result_dt_mallows.txt","a")
    file.write("dataset: nytimes\n")
    file.write("CV: %d\n" % 5)
    file.write(str(result)+"\n")
    file.close()
