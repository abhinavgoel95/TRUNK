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


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

HNN = hierarchy.Hierarchy(args.pretrained_path)
testset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

process_data(testloader)




