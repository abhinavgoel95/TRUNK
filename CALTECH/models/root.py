import torch.nn as nn

cfg = {
    'VGG11': [64, 128, 'M', 256, 'D', 256, 'M', 256, 256, 'M', 'D'],
    '4': [16, 32, 'M', 32, 32, 'M', 32, 32], #90.9
    '5': [16, 32, 'M', 32, 'M', 32, 64], #92.9
    '6': [16, 32, 'M', 32, 32, 'M', 32],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_down = {
'4': ['M', 32, 64, 'M', 64, 128, 'M'],
'5': ['M', 64, 64, 'M', 128, 128, 'M'],
'6': ['M', 64, 'M', 64, 128, 'M'],
}
class model(nn.Module):
    def __init__(self, vgg_name):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.features_down = self._make_layers(cfg_down[vgg_name], inc = 32)
        self.classifier = nn.Sequential(
                        nn.Linear(7*7*128, 7),
                )

    def forward(self, x):
        y = self.features(x)
        x = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return y, out

    def _make_layers(self, cfg, inc = 3):
        layers = []
        in_channels = inc
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def evaluate(self, data, target, device):
        self.eval()
        data = data.to(device)
        target = target[0]
        target = target.to(device)
        y, net_out = self(data)
        return y, net_out

def get_root(size):
    return model(size)

