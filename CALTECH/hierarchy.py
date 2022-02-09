from sklearn.decomposition import non_negative_factorization
import torch
import models
import os

class Hierarchy:
    def __init__(self, pretrained_path = './trained_weights'):
        
        self.nodes = ['root', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5', 'SG6', 'SG7', 'SG8', 'SG9']
        self.valid_paths = [
            (0, 0, 0 ),
            (1, 0, 0 ),
            (2, 0, 0, 0),
            (0, 0, 1 ),
            (3, 0 ),
            (0, 0, 2 ),
            (2, 1, 0 ),
            (1, 0, 1 ),
            (4,  ),
            (5,   ),
            (0, 1  ),
            (3, 1  ),
            (2, 2  ),
            (2, 0, 0, 1),
            (2, 0, 1 ),
            (6,   ),
            (1, 0, 2 ),
            (1, 1  ),
            (0, 2  ),
            (2, 1, 1 ),
        ]
        self.leaves = ('cannon','cd','monitor','dog','dolphin','duck','jet','frisbee','harp','harpsichord', 'whale', 'laptop', 'microwave', 'refrigerator', 'school bus', 'soccer ball', 'tennis ball', 'bike', 'washing machine')
        self.pretrained_path = pretrained_path

        self.next = {
            'root': ['SG1', 'SG2', 'SG3', 'SG6', None, None, None],
            'SG1': ['SG4', None, None] ,
            'SG2': ['SG5', None],
            'SG3': ['SG7', 'SG8', None],
            'SG7': ['SG9', None],
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