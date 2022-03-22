import numpy as np


class OpenSeesNodeOutput:
    def __init__(self, filename, ncomps, nodeTags=[], compNames=[], **kwargs):
        """
        Read opensees generated node output file.
        Inputs:
        filename (string): Node output filename.
        ncomps (int): number of components per node.
        nodeTags (list of int): Node tags, default=[1, 2, ...., N]. Note 
        that default list starts from "1" for compatibility with opensees.
        compnames (list of strings): Name to be given to each component. 
        Default is ["0", "1", "2", "3", ...., "m"]
        """
        with open(filename, "r") as f:
            data = np.genfromtxt(filename)

        nNodes = int((np.shape(data)[1] - 1)/ncomps)
        if len(nodeTags) == 0:
            nodeTags = [i+1 for i in range(nNodes)]
        elif len(nodeTags) != nNodes:
            raise Exception("Length of nodeTags and number of nodes must be same")

        if len(compNames) == 0:
            compNames = ["{}".format(i) for i in range(ncomps)]

        self.t = data[:, 0]
        d = np.split(data[:, 1::], nNodes, axis=1)
        dataDir = {}
        for i in nodeTags:
            dataDir[i] = {}
            for j in range(ncomps):
                dataDir[i][compNames[j]] = d[i-1][:, j]

        self.data = dataDir
        self.nodeTags = nodeTags
        self.compNames = compNames
