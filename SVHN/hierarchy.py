from sklearn.decomposition import non_negative_factorization
import torch
import models
import os

class Hierarchy:
    """
        Defining the structure of the hierarchy used for the Extended MNIST Dataset
    """
    def __init__(self, pretrained_path = './trained_weights'):
        
        self.nodes = ['root', 'SG1', 'SG2']
        self.valid_paths = [
            (0,), 
            (1,0), 
            (2,), 
            (3,0), 
            (4,), 
            (3,1), 
            (3,3), 
            (1,1), 
            (3,2), 
            (5,)
        ]
        self.leaves = (0,1,2,3,4,5,6,7,8,9)
        self.pretrained_path = pretrained_path

        self.next = {
            'root': [None, 'SG1', None, 'SG2', None, None]
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