"""
data preprocessing
"""

import numpy as np
import ast


#######################################################################
######### read Facebook data ###########
emoticon_list = ["Like","Love","Sad","Wow","Haha","Angry"] # emoticon list for Facebook data to transform label to index

def dataFacebook(datafile):
    file = open(datafile,"r")
    x=[]
    y=[]
    for item in file.readlines():
        try:
            sample = ast.literal_eval(item.rstrip())
        except SyntaxError, e:
            print e.message
            continue
        if sample["feature_emotion"][0]<0:
            continue
        emoticons = [0 for i in range(len(emoticon_list))]
        for j in range(len(emoticon_list)):
            if emoticon_list[j] in sample["emoticons"].keys():
                emoticons[j] = sample["emoticons"][emoticon_list[j]]
            else:
                emoticons[j] = 0
        flag_withemoticon = False
        for j in range(len(emoticon_list)):
            if emoticons[j]>0:
                flag_withemoticon = True
        if flag_withemoticon:
            x.append(sample["feature_emotion"])
            y.append(emoticons)
    file.close()
    x = np.asarray(x,dtype="float")
    y = np.asarray(y,dtype="float")
    return x,y

##############################################################
## transforming votes data to ranking data for Facebook data ##
def label2Rank(y):
    y = map(rankOrder, y)
    y = np.array(y, dtype=int)
    return y


def rankOrder(dist):
    # output rank[i] = j ( means j is the index of item ranking i)
    rank=[i for i in range(len(dist))]
    for i in range(1,len(dist)):
        for j in range(1,len(dist)-i+1):
            if dist[rank[j]]>dist[rank[j-1]]:
                temp=rank[j]
                rank[j]=rank[j-1]
                rank[j-1]=temp
    for i in range(len(dist)):
        if dist[rank[i]]==0:
            rank[i]=-1
    return rank
###############################################################
###############################################################

###############################################################
######### read semi-synthetic data ############################
def readSyntheticData(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        for line in f.readlines():
            atts = line.rstrip("\n").split(",")
            y_single = atts[-1]
            del atts[-1]
            x_single = map(float, atts)
            x.append(x_single)
            y.append(rankParse(y_single))
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.int16) - 1 # minus 1 for 0-index
    return x, y

def rankParse(rankstr):
    labels = rankstr.split(">")
    labels = map(lambda x: int(x.lstrip("L")), labels)
    return labels

