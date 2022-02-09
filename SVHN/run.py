import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
from torchvision import datasets, transforms
import hierarchy


def process_data(testloader):
    for frame_number, (image, label) in enumerate(testloader):
        current_DNN = 'root'
        path = []
        while current_DNN != None:
            model = HNN.getModel(current_DNN)
            if torch.cuda.is_available():
                model = model.cuda()
                image = image.cuda()
                label = label.cuda()
            model.eval()
            image, net_out = model(image)
            output = net_out.max(1, keepdim=True)[1].item()
            path.append(output)
            current_DNN = HNN.getNext(current_DNN, output)
            del model
        print('frame number: ', frame_number, ' label: ', HNN.getLeaf(path), ' target: ', HNN.getTarget(path), ' ground truth: ', label.item())


parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR', default = './data', help='path to testing dataset')
parser.add_argument('--pretrained_path', metavar='DIR', default = './trained_weights', help='path to testing dataset')
args = parser.parse_args()


testset = datasets.SVHN(root=args.data, split = 'test', download=True, transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, shuffle = True, batch_size = 1)

HNN = hierarchy.Hierarchy(args.pretrained_path)


process_data(testloader)




