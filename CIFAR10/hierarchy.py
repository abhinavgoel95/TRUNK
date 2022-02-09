import torch
import models
import os

class Hierarchy:
    """
        Defining the structure of the hierarchy used for the Extended MNIST Dataset
    """
    def __init__(self, pretrained_path):
        
        self.nodes = ['root', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5']
        self.valid_paths = [(0,0), (1,0), (2,0,0), (2,1,0), (2,0,1), (2,1,1), (2,2), (2,3), (0,1), (1,1)]
        self.leaves = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.pretrained_path = pretrained_path

        self.next = {
            'root': ['SG1', 'SG2', 'SG3'],
            'SG1': [None, None],
            'SG2': [None, None],
            'SG3': ['SG4', 'SG5', None, None],
            'SG4': [None, None],
            'SG5': [None, None],
        }

        self.path_to_leaf = dict(
            zip(
                self.valid_paths,
                self.leaves
            )
        )

        self.path_to_target = dict(
            zip(
                self.valid_paths, 
                list(range(10))
            )
        )

    def getNext(self, current_node, child):
        next_child = self.next[current_node][child]
        if next_child == None:
            return None
        return next_child
    
    def getLeaf(self, path):
        path = tuple(path)
        if path not in self.path_to_leaf:
            return None
        return self.path_to_leaf[path]

    def getTarget(self, path):
        path = tuple(path)
        if path not in self.path_to_target:
            return None
        return self.path_to_target[path]

    def getModel(self, DNN_name):
        model = getattr(models, "get_"+DNN_name+"_model")()
        path = os.path.join(self.pretrained_path,DNN_name+'.pth')
        model.load_state_dict(torch.load(path))
        return model