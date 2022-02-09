from sklearn.decomposition import non_negative_factorization
import torch
import models
import os

class Hierarchy:
    """
        Defining the structure of the hierarchy used for the Extended MNIST Dataset
    """
    def __init__(self, pretrained_path = './trained_weights'):
        
        self.nodes = ['root', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5', 'SG6', 'SG7', 'SG8', 'SG9', 'SG10', 'SG11', 'SG12', 'SG13']
        self.valid_paths = [
            (0,0,0),
            (1,0,0),
            (0,3,0),
            (2,),
            (3,0), 
            (4,0),
            (3,11),
            (5,),
            (0,6),
            (0,5,2),
            (3,4),
            (0,7),
            (6,0),
            (0,1),
            (7,),
            (8,0,0),
            (3,12),
            (3,1),
            (1,0,1),
            (3,7),
            (1,1),
            (1,0,2),
            (9,0),
            (10,0),
            (0,0,1),
            (11,),
            (0,4),
            (3,8),
            (4,1),
            (8,2),
            (3,2),
            (3,5),
            (10,1),
            (3,6),
            (3,0),
            (0,3,1),
            (0,2),
            (3,10),
            (12,),
            (6,1),
            (8,0,1),
            (0,5,1),
            (3,9),
            (9,1),
            (0,5,0),
            (13,), 
            (8,1),
        ]
        self.leaves = (0,1,2,3,4,5,6,7,8,9,'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'n', 'q', 'r', 't')
        self.pretrained_path = pretrained_path

        self.next = {
            'root': ['SG1', 'SG2', None, 'SG3', 'SG4', None, 'SG5', None, 'SG6', 'SG7', 'SG8', None, None, None ],
            'SG1': ['SG9', None, None, 'SG10', None, 'SG11', None, None],
            'SG2': ['SG12', None],
            'SG6': ['SG13', None, None],
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
                list(range(len(self.valid_paths)))
            )
        )

    def getNext(self, current_node, child):
        if current_node not in self.next:
            return None

        next_child = self.next[current_node][child]
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