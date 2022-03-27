import numpy as np
import json


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

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:\n".format(key)
        return a


class OpenSeesModel:
    def __init__(self, jsonModelFile):
        """
        Class for storing OpenSees Model details recorded in a JSON file using command "print -JSON -file jsonModelFile
        """
        self.jsonFile = jsonModelFile

        # Parse JSON
        self.json_parser()

    def __repr__(self):
        b = self.__str__()
        return b

    def __str__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:\n".format(key)
        return a

    def json_parser(self):
        # Read file
        with open(self.jsonFile, "r") as f:
            data = f.readlines()

        nodes = {}
        elements = {}
        for line in data:
            line = line.strip("\t|\n|,")
            if ("name" in line) and ("crd" in line):
                j = json.loads(line)
                nodes = {**nodes, **{j["name"]: j["crd"]}}
            elif ("name" in line) and ("nodes" in line):
                j = json.loads(line)
                elements = {**elements, **{j["name"]: {"type": j["type"], "nodes": j["nodes"], "material": j["material"]}}}
        self.nodes = nodes
        self.elements = elements
