import numpy as np
import math

from DecisionTreeWeight import crossValidate
from DecisionTreeMallows import crossValidate
from logRegression import crossValidate
from RankPairPref import crossValidate
from LabelWiseRanking import crossValidate
from KNNPlackettLuce import crossValidate
from KNNMallows import crossValidate
from logLinear import crossValidate
from SMPrank import crossValidate

import ReadData

################# read data ####################
## for Facebook data ##
x, y = ReadData.dataFacebook("data/nytimes.txt")
y = ReadData.label2Rank(y)  # transform to label ranking data ! skip it when using LogR and NAIVE

## for semi-synthetic data ##
x, y = ReadData.readSyntheticData("data/synthetic/bodyfat") # label ranking data, not supporting LogR and NAIVE

################################################

################# apply algos ##################
## DecisionTreeWeight (DTPG) ##
results = crossValidate(x, y, cv=5, alpha=0.0, rank_weight=False, stop_criterion_mis_rate=None, stop_criterion_min_node=1,
              stop_criterion_gain=0.0, prune_criteria=0)
# alpha = None for searching best pruning hyperparameter in training data;
# prune_criteria decides which of the performance measure is used to choose hyperparameter in pruning

## DecisionTreeMallows (LRT) ##
results = crossValidate(x,y, method = "dT",cv=5, alpha = 0.0, min_node = 1)

## LogRegression ##
## LogR ##
results = crossValidate(x,y ,cv=5)
## NAIVE ##
x_naive = np.ones([x.shape[0],1]).astype("float") # uniform features
results = crossValidate(x_naive, y, cv=5)

## RankPairPref (RPC) ##
results = crossValidate(x,y,cv=5, k=2, Abstention = False)

## logLinear (LogLinear) ##
results = crossValidate(x, y, cv=5)

## SMPrank (SMP) ##
"""
K = min(max(math.factorial(y.shape[1] - 1), math.factorial(y.shape[1]) / 2), y.shape[0] / 50)
if synthetic/elevators
    K = 50
if Facebook data
    K = 100
"""
results = crossValidate(x, y, cv=5, K=100)

## LabelWiseRanking (LWR) ##
results = crossValidate(x,y,cv=5)

## KNNPlackettLuce (KNN-PL) & KNNMallows (KNN-M) ##
results = crossValidate(x, y, cv=5, K=20)   # K: KNN

##############################################################