"""
Logistic regression (LogR) and NAIVE baselines for label ranking in votes data (original Facebook data)
NAIVE is the special case of LogR, as input features are set to uniform
LogR is essentially a multi-class classification algorithm with probability distribution output
"""

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np

from PerfMeasure import perfMeasure
import ReadData

def logRegFeatureEmotion(x_training,y_training, x_test):
    logReg = linear_model.LogisticRegression(C=1e9, fit_intercept=True, multi_class="multinomial", solver="newton-cg")### test
    fitResult = logReg.fit(x_training,y_training)
    y= fitResult.predict_proba(x_test)
    coef = logReg.coef_
    intercept = logReg.intercept_
    return y, coef, intercept

def multiClass(x,y):
    # reducing multiclass samples with cumulative # labels to samples each with one label
    y_shape = y.shape
    n_sample = y_shape[0]
    n_class = y_shape[1]

    rep = np.sum(y, axis = 1)
    x_rep = np.repeat(x,rep.astype(int),axis=0)

    base_class = np.array([i for i in range(n_class)])
    y_rep = np.repeat(base_class.reshape([1,n_class]),n_sample,axis = 0)
    y_rep = y_rep.reshape([n_class*n_sample])
    rep = y.reshape([n_class*n_sample])
    y_rep = np.repeat(y_rep,rep.astype(int))

    return x_rep, y_rep


def crossValidate(x,y ,cv=5):

    results = {"perf":[], "coef":[], "interc":[]}
    # cross validation #
    np.random.seed(1100)
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
    for train,test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        # from multilabel to multiclass based on independencec assumption
        x_train, y_train = multiClass(x_train,y_train)

        # feature standardization #
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # training and predict
        result = logRegFeatureEmotion(x_train, y_train, x_test)

        # performance measure
        y_pred, coef, interc = result
        results["perf"].append(perfMeasure(y_pred,y_test))
        results["coef"].append(coef)
        results["interc"].append(interc)

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        results[key] = [mean, std]

    return results

if __name__ == "__main__":
    x,y = ReadData.dataFacebook("data/nytimes.txt")
    results = crossValidate(x[:100,:],y[:100,:])
    print results