#Program used to visualize decision trees. Outputs pngs

import pydotplus
import collections

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


