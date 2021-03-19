import torch.nn as nn

__all__ = ['hynet']


cfg = {
    '24642': [32, 64, 'M', \
              64, 64, 96, 128, 'M', \
              128, 128, 160, 160, 192, 256, 'M', \
              256, 256, 256, 512, 'M', \
              512]
}


class Hynet(nn.Module):
    def __init__(self, base, feature_dim=256):
        super(Hynet, self).__init__()
        # TODO: enable different types of fully-connected
        self.base = base
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256 * 7 * 7, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    out_channels = 256
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.Conv2d(in_channels, out_channels, 1, bias=False), \
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def hynet(**kwargs):
    """Constructs a hynet model.
    """
    model = Hynet(make_layers(cfg['24642']), **kwargs)
    return model
