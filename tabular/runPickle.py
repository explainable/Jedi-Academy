import pickle
from sklearn.datasets import load_boston

__dir__ = "/".join(__file__.rsplit('/')[:-1])

def readInTitanic():
    allInsts = []
    allLabels = []
    featNames = []
    with open(__dir__ + "/data/titanic.csv") as titFile:
        for linenum, line in enumerate(titFile):
            features = line.split(",")
            currInst = []
            for index, feat in enumerate(features):
                if linenum == 0:
                    if index != 0 and index != 2:
                        featNames.append(feat.replace('\r', '').replace('\n',''))
                else: 
                    if index == 0:
                        allLabels.append(int(feat))
                    elif index != 2:
                        if feat == "male":
                            currInst.append(0.0)
                        elif feat == "female":
                            currInst.append(1.0)
                        else:
                            currInst.append(float(feat))
            if linenum != 0:
                allInsts.append(currInst)

    return allInsts, allLabels, featNames

def loadTitanicTree():
    loaded_clf = pickle.load(open(__dir__ + "/data/titanicTree.pkl", 'rb'))
    features, labels, featNames = readInTitanic()
    deciPath = loaded_clf.decision_path(features).toarray()
    
    return loaded_clf, featNames, deciPath

def loadBostonTree():
    loaded_clf = pickle.load(open(__dir__ + "/data/bostonTree.pkl", 'rb'))

    boston = load_boston()
    features, labels = load_boston(True)
    data_feature_names = boston.feature_names

    deciPath = loaded_clf.decision_path(features).toarray()
    
    return loaded_clf, data_feature_names, deciPath



#loadTitanicTree()
#loadBostonTree()
