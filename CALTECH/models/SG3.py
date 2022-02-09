import torch.nn as nn


cfg = {
    '1': [128, 'M'], #65
    '2': [64, 'M', 64, 64, 'M'],
    '3': [32, 32, 32, 'M', 32, 64, 64, 'M'],
    '4': [16, 32, 32, 32, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64,'D'],
    '6': [16, 32, 32, 'M', 64, 64, 128, 'M', 'D'],
}


cfg_down = {
    '2': [64, 128, 'M'],
    '3': [64, 64, 64, 128, 'M'],
}

class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[size], channels = 32)
        self.features_down = self._make_layers(cfg_down[size], channels = 64)
        self.classifier = nn.Sequential(
                        nn.Linear(128*7*7, 3),
                )

    def forward(self, x):
        y = self.features(x)
        x = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return y, out

    def _make_layers(self, cfg, channels = 32):
        layers = []
        in_channels = channels
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


def get_SG3(size):
    return model(size)
