import torch.nn as nn

cfg = {
    '4': [32, 64, 'M'],
}

class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(64*3*3, 3),
                )

    def forward(self, x):
        y = self.features(x)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg, channels = 3):
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
        y, net_out = self(data)
        return y, net_out


def get_SG12(size):
    return model('4')