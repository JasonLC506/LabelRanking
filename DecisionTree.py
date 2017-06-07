import numpy as np
from sklearn.model_selection import KFold
from functools import partial
from scipy.stats.mstats import gmean
from datetime import datetime

MINNODE = 1

def divideset(x,samples, feature, value):
    # divide samples (index) into two sets according to split value

    if type(x) != np.ndarray:
        x= np.array(x)
    if type(samples)!= np.ndarray:
        samples = np.array(samples)
    split_function = None
    if isinstance(value,int) or isinstance(value,float):
        # numerical feature
        split_function = lambda sample: x[sample,feature]>=value
    else:
        raise("nominal feature not supported")

    index = split_function(samples)
    set1 = samples[index]
    set2 = samples[index==False]
    return set1, set2


def giniRank(y,samples):
    # y should be ranking data here
    # calculate it for every split node, which is O(n^2), can do better in O(nlogn)
    if type(y) != np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    Nclass = Ranks
    gini_rank = 0.0
    for rank in range(Ranks):
        n_class = [0.0 for i in range(Nclass)]
        gini = 0.0
        for sample in samples:
            emoti = int(y[sample,rank])
            if emoti>=0:
                n_class[emoti]+=1
        n = sum(n_class)
        if n < 1:
            gini_rank += gini*n
        else:
            gini = sum([n_class[i]*(n-n_class[i]) for i in range(Nclass)])*1.0/n/n
            gini_rank += gini * n # adding weight of size
        # ### test ###
        # print "n_class: ", n_class
        # print "gini: ", gini
    return gini_rank


def bestSplit(x, y, samples, feature, min_node = 1):
    min_gini = -1
    best_split = 0
    best_sets = []

    Nsamp = len(samples)
    Ranks = y.shape[1]
    Nclass = Ranks

    temp = [(x[samples[i],feature],samples[i]) for i in range(len(samples))]
    dtype = [("value", float),("index", int)]
    x_ord = np.sort(np.array(temp,dtype=dtype), order="value")

    n_rc = [[[0.0 for i in range(Nclass)] for j in range(Ranks)] for i in range(2)]
    # n_rc[0] results for tb; n_rc[1] results for fb

    n_rc[0] = nRankClass(y,samples)

    j = 0
    old_value = x_ord[0][0]
    for i in range(Nsamp-1):
        value = x_ord[i][0]
        if value == old_value:
            n_rc[0] = nRankClassDel(n_rc[0],y[x_ord[i][1],:])
            n_rc[1] = nRankClassAdd(n_rc[1],y[x_ord[i][1],:])
            if x_ord[i+1][0] > value:
                j=i+1
                old_value = x_ord[i+1][0]
                gini_tb = giniRank_e(n_rc[0])
                gini_fb = giniRank_e(n_rc[1])
                gini = gini_tb + gini_fb
                # print "current gini", gini_tb, gini_fb
                # print "current sets", [[y[n,:] for n in range(j,Nsamp)], [y[m,:] for m in range(j)]]
                if min_gini < 0 or min_gini >= gini:
                    min_gini = gini
                    best_split = j

    best_sets = [[x_ord[i][1] for i in range(best_split,Nsamp)], [x_ord[j][1]for j in range(best_split)]]
    best_split = [feature, x_ord[best_split][0]]

    return min_gini, best_split, best_sets


def giniRank_e(n_rc):
    Ranks = len(n_rc)
    Nclass = len(n_rc[0])
    gini_rank = 0.0
    for rank in range(Ranks):
        gini = 0.0
        n = sum(n_rc[rank])
        if n < 1:
            gini_rank +=gini*n
        else:
            gini = sum([n_rc[rank][i]*(n-n_rc[rank][i]) for i in range(Nclass)]) *1.0/n/n
            gini_rank += gini*n
    return gini_rank


def nRankClassDel(n_rc, y_rank):
    Ranks = len(n_rc)
    Nclass = len(n_rc[0])
    for rank in range(Ranks):
        emoti = int(y_rank[rank])
        if emoti<0:
            break
        n_rc[rank][emoti] = n_rc[rank][emoti] - 1
        if n_rc[rank][emoti]<0:
            print "wrong delete"
    return n_rc

def nRankClassAdd(n_rc,y_rank):
    Ranks = len(n_rc)
    for rank in range(Ranks):
        emoti = int(y_rank[rank])
        if emoti < 0:
            break
        n_rc[rank][emoti] += 1
    return n_rc

def nRankClass(y,samples):
    if type(y) != np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    Nclass = Ranks
    n_rc = [[0.0 for i in range(Nclass)] for j in range(Ranks)]
    for rank in range(Ranks):
        for sample in samples:
            emoti = int(y[sample,rank])
            if emoti>=0:
                n_rc[rank][emoti]+=1
    return n_rc


# def rankResult(y,samples):
#     if type(y) != np.ndarray:
#         y = np.array(y)
#     Ranks = y.shape[1]
#     N_class = Ranks
#     result = []
#     n_class =[[] for i in range(N_class)]
#     for rank in range(N_class):
#         n_class[rank] = [0 for i in range(N_class)]
#         for sample in samples:
#             emoti = y[sample,rank]
#             if emoti>=0:
#                 n_class[rank][emoti]+=1
#         # assign ranking #
#         if sum(n_class[rank])==0:
#             result.append(-1)
#             continue
#         flag = False
#         while flag == False:
#             max_value = max(n_class[rank])
#             for i in range(N_class):
#                 if n_class[rank][i] == max_value:
#                     if i not in result:
#                         result.append(i)
#                         flag = True
#                     else:
#                         n_class[rank][i] = 0
#     return result


def rankResult(y,samples):
    if type(y)!=np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    N_class = Ranks
    result = []
    n_class =[[] for i in range(N_class)]
    for rank in range(N_class):
        n_class[rank] = [0 for i in range(N_class)]
        for sample in samples:
            emoti = y[sample,rank]
            if emoti>=0:
                n_class[rank][emoti]+=1
    for rank in range(N_class):
        # assign ranking #
        # current rank with highest priority, when zero appearance, from highest rank to lowest #
        priority = [i for i in range(N_class) if i != rank]
        priority.insert(0,rank)
        flag = False
        for i in priority:
            n_class_cur = n_class[i]
            while not flag:
                max_value = max(n_class_cur)
                if max_value < 1:
                    break # all are 0 in current rank
                emoti = n_class_cur.index(max_value)
                if emoti not in result:
                    result.append(emoti)
                    flag = True # find best emoticon
                else:
                    n_class_cur[emoti]=0
            if flag:
                break
        if not flag:
            result.append(-1)
    return result


class decisionnode:
    def __init__(self,feature=-1,value=None, result = None, tb=None, fb=None, gain=0.0, size_subtree = 1):
        self.feature = feature
        self.value = value
        self.result = result
        self.tb = tb
        self.fb = fb
        # pruning #
        self.gain = gain
        self.size = size_subtree
        self.alpha = -1.0
        if self.size > 1:
            self.alpha = gain/(self.size-1)


def buildtree(x,y, samples, min_node=1):
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples) != np.ndarray:
        samples = np.array(samples)
    if len(samples) == 0:
        return decisionnode()
    current_criterion = giniRank_e(nRankClass(y,samples))

    if len(samples)<= min_node:
        return decisionnode(result=rankResult(y,samples))
    # find best split
    best_gain = 0.0
    best_split = []
    best_sets = []

    N_feature = x.shape[1]

    for feature in range(N_feature):
        # nlogn selection
        best_gini, split, sets = bestSplit(x,y,samples,feature)
        gain = current_criterion - best_gini
        if gain > best_gain and len(sets[0]) * len(sets[1]) > 0:
            best_gain = gain
            best_split = split
            best_sets = sets

    if best_gain>0:
        tb = buildtree(x,y, best_sets[0], min_node = min_node)
        fb = buildtree(x,y, best_sets[1], min_node = min_node)
        return decisionnode(feature = best_split[0], value = best_split[1], result = rankResult(y,samples),
                            tb = tb, fb = fb,
                            gain = (tb.gain+fb.gain+best_gain), size_subtree = (tb.size+fb.size))
    else:
        return decisionnode(result = rankResult(y,samples))


def prune(tree, alpha):
    if tree.tb == None:
        return decisionnode(result = tree.result)
    else:
        if tree.alpha >= alpha:
            tb = prune(tree.tb, alpha)
            fb = prune(tree.fb, alpha)
            return decisionnode(feature = tree.feature, value = tree.value, result = tree.result,
                                tb = tb, fb = fb)
        else:
            return decisionnode(result = tree.result)


def printtree(tree,indent=""):
    if tree.tb == None:
        print(indent+str(tree.result))
    else:
        print(indent+str(tree.feature)+">="+str(tree.value)+"?")
        print(indent+"T->\n")
        printtree(tree.tb,indent+"  ")
        print(indent + "F->\n")
        printtree(tree.fb,indent+"  ")

def alphaList(tree,alpha_list):
    if tree.tb != None:
        alpha_list.insert(tree.alpha)
        alphaList(tree.tb, alpha_list)
        alphaList(tree.fb, alpha_list)
    else:
        pass


class orderList:
    def __init__(self):
        self.list = []

    def insert(self, x):
        L = len(self.list)
        for i in range(L):
            if self.list[i] > x:
                self.list.insert(i,x)
                return self
            elif self.list[i] == x:
                return self
        self.list.append(x)

    def printAll(self):
        print self.list

    def length(self):
        return len(self.list)


def buildPruneList(x,y):
    # x: Nsamp*Nfeature; y: Nsamp*Nrank
    samples = [i for i in range(x.shape[0])]
    tree_max = buildtree(x,y,samples)
    alpha_list = orderList()    # initialize
    alphaList(tree_max, alpha_list)
    PruneList = {}
    # insert the largest alpha to include root tree #
    alpha_max = max(alpha_list.list) + 1
    alpha_list.insert(alpha_max)
    for alpha in alpha_list.list:
        PruneList[alpha] = prune(tree_max,alpha)

    return PruneList


def predict(observation,tree, alpha):
    # prediction of single observation
    if tree.tb == None:
        return tree.result
    if tree.alpha >= 0:
        if tree.alpha < alpha:
            return tree.result
    value = observation[tree.feature]
    if isinstance(value,int) or isinstance(value,float):
        if value>=tree.value:
            branch = tree.tb
        else:
            branch = tree.fb
    else:
        raise("nominal feature not supported")
    return predict(observation,branch,alpha)


def decisionTree(x_train, y_train, x_test, alpha = None):
    # x: Nsamp*Nfeature, y: Nsamp*Nrank ranking data
    # prune_list = buildPruneList(x_train, y_train)
    # alpha_list = prune_list.keys()
    # alpha_list.sort()
    # print "start building tree: ", datetime.now() ### test
    tree = buildtree(x_train, y_train, [i for i in range(x_train.shape[0])])
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


def hyperParometer(x, y, cv = 5, alpha_try_list = np.linspace(0.0,5.0,10)):
    # select pruning hyperparometer by cross validation #
    ntrial = len(alpha_try_list)
    perf = [0.0 for i in range(ntrial)]
    # cross validation #
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
    for train,test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]
        # print "start building prediction: ", datetime.now() ### test
        alpha_list, y_pred_list = decisionTree(x_train, y_train, x_test)
        # print "end building prediction: ", datetime.now() ### test
        Nalpha = len(alpha_list)
        performance = [GMean(y_pred_list[i],y_test) for i in range(Nalpha)]
        # assign performance value to each alpha_try #
        for trial in range(ntrial):
            alpha_try = alpha_try_list[trial]
            overlarge = True
            for ind in range(Nalpha-1):
                if alpha_try <= alpha_list[ind]:
                    perf[trial] += performance[ind]
                    overlarge = False
                    break
            if overlarge:
                perf[trial] += performance[Nalpha-1]
    # find best performance alpha and perf value #
    max_perf = max(perf)
    max_ind = None
    for trial in range(ntrial):
        ind = ntrial-1-trial    # prefer larger alpha in same perf
        if perf[ind] == max_perf:
            max_ind = ind
            break
    return alpha_try_list[max_ind], perf[max_ind]*1.0/cv


def GMean(y_pred, y_test):
    recall, Nsamp_class = LogR.recallSub(y_pred, y_test)
    recall = np.array(recall,dtype="float")
    g_mean = gmean(recall)
    return g_mean

def selectTrainTest(x, y, frac = np.array([2.0,1.0,1.0]), cv_select=5, cv_test=5, select_crt = GMean):
    Nsamps = x.shape[0]
    frac_train = int(frac[0]/np.sum(frac)*Nsamps)
    frac_valid = int(frac[1]/np.sum(frac)*Nsamps)
    frac_test = int(frac[2]/np.sum(frac)*Nsamps)
    pass


def dataSimulated(Nsamp, Nfeature, Nclass):
    np.random.seed(seed=10)
    x = np.arange(Nsamp*Nfeature,dtype="float").reshape([Nsamp,Nfeature])
    x += np.random.random(x.shape)*10
    y = np.random.random(Nsamp*Nclass).reshape([Nsamp, Nclass])
    y *= 2
    y = y.astype(int)
    y = map(LogR.rankOrder,y)
    y = np.array(y)
    return x,y


def label2Rank(y):
    y = map(LogR.rankOrder,y)
    y = np.array(y, dtype=int)
    return y


