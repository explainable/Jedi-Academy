#Program used to visualize decision trees. Outputs pngs

import pydotplus
import collections
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#Takes dot_data, a string describing the nodes and edges of a given decision
#tree. This string is returned by a treeReg or treeClassify model in the
#model.py file. Additionally, it takes filename, the name of the file where the
#image will be saved
#
#This function returns a graphical representation of a decision tree in the
#form of a png
def plotTree(dot_data, filename):
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(filename)

def convertTreeToParentChild(clf, featNames, predLabels, deciPath):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    parents = [""]
    labels = ["0: Everyone"]

    numLive = 0
    numDead = 0
    for pred in predLabels:
        if pred == 1:
            numLive += 1
        if pred == 0:
            numDead += 1
    initVal = "0: Dead = %d%%, Live = %d%%" % \
                                (numDead * 100 / float(numDead + numLive), \
                                (numLive * 100 / float(numDead + numLive)))
    initVal = numLive / float(numDead + numLive)

    vals = [initVal]

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    leafId = 0
    nodeVals = {0: ""}
    for i in range(n_nodes):
        if not is_leaves[i]:
            numLive = 0
            numDead = 0
            for index, pred in enumerate(predLabels):
                if deciPath[index][children_left[i]] == 1:
                    if pred == 1:
                        numLive += 1
                    else:
                        numDead += 1

            leftVal = "Dead = %d%%, Live = %d%%" % \
                                (numDead * 100 / float(numDead + numLive), \
                                (numLive * 100 / float(numDead + numLive)))
            leftVal = numLive / float(numDead + numLive)
            #leftVal = str(children_left[i]) + ": " + leftVal
            leftLabel = featNames[feature[i]] + "<=" + str(threshold[i])
            leftLabel = str(children_left[i]) + ": " + leftLabel
            labels.append(leftLabel)
            vals.append(leftVal)
            nodeVals[children_left[i]] = leftLabel
            parents.append(nodeVals[i])


            numLive = 0
            numDead = 0
            for index, pred in enumerate(predLabels):
                if deciPath[index][children_right[i]] == 1:
                    if pred == 1:
                        numLive += 1
                    else:
                        numDead += 1
            rightVal = "Dead = %d%%, Live = %d%%" % \
                                (numDead * 100 / float(numDead + numLive), \
                                (numLive * 100 / float(numDead + numLive)))
            rightVal = numLive / float(numDead + numLive)
            #rightVal = str(children_right[i]) + ": " + rightVal

            rightLabel = featNames[feature[i]] + ">" + str(threshold[i])
            rightLabel = str(children_right[i]) + ": " + rightLabel

            vals.append(rightVal)
            labels.append(rightLabel)
            nodeVals[children_right[i]] = rightLabel
            parents.append(nodeVals[i])

    print parents
    print labels
    print vals

def convertDotData(dot_data):
    graph = pydotplus.graph_from_dot_data(dot_data)

    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    semiSplit = dot_data.split(";")
    nodeSplit = []
    for entry in semiSplit:
        entrySplit = entry.split()
        valid = True
        for ent in entrySplit:
            if ent == "->":
                valid = False

        if valid:
            nodeSplit.append(entry)

    nodes = []
    for node in nodeSplit:
        nodes.append(node.split('\n')[1])

    print edges
    nodeInfo = {}
    for index, node in enumerate(nodes):
        if index != 0 and index != 1 and index != len(nodes) - 1:
            thisNode = node.decode()
            front = thisNode.split("\\")[0]
            twoParts = front.split("\"")
            num = twoParts[0].split()[0]
            label = twoParts[1]
            nodeInfo[str(num)] = str(label)
            
    print nodeInfo
