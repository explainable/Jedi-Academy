import pickle

def loadTitanicTree():
    loaded_clf = pickle.load(open("titanicTree.pkl", 'rb'))
    #print(loaded_clf.predict([[3, 1, 22, 1, 0, 7.25]]))

#loadTitanicTree()
