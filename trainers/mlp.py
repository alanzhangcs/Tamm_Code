import torch.nn as nn


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activate="gelu", drop=0.2):
        super().__init__()
        self.act = get_activation(activate)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            self.act,
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            self.act,
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.classifier(x)


import MinkowskiEngine as ME


class MLP_ME(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.final = nn.Sequential(
            self.get_mlp_block(in_features, hidden_features),
            ME.MinkowskiDropout(0.5),
            self.get_mlp_block(hidden_features, out_features),
            ME.MinkowskiDropout(0.5),
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiGELU(),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        return self.final(x).F


