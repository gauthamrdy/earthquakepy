from .opensees_classes import OpenSeesNodeOutput, OpenSeesModel


def read_ops_json_model(filename):
    """
    Reads opensees generated json model file obtained using following command:
    print -JSON -file filename
    """
    return OpenSeesModel(filename)


def read_ops_node_output(filename, ncomps, nodeTags=[], compNames=[], **kwargs):
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
    return OpenSeesNodeOutput(filename, ncomps, nodeTags, compNames, **kwargs)


def read_ops_element_output(filename, ncomps, elmTags=[], compNames=[], **kwarga):
    """
    Read opensees generated element output file.
    Inputs:
    filename (string): Element output filename.
    ncomps (int): number of components per element.
    elmTags (list of int): Element tags, default=[1, 2, ...., N]. Note 
    that default list starts from "1" for compatibility with opensees.
    compnames (list of strings): Name to be given to each component. 
    Default is ["0", "1", "2", "3", ...., "m"]
    """
    return OpenSeesNodeOutput(filename, ncomps, elmTags, compNames, **kwargs)
