import torch
import torch.nn as nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class NewCLIP(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(NewCLIP, self).__init__()
        self.adapter = Adapter(c_in, reduction=reduction)
        self.ratio = ratio

    def forward(self, image_features):
        x = self.adapter(image_features)
        ratio = self.ratio
        image_features = ratio * x + (1 - ratio) * image_features

        return image_features



def make(cfg):

    # if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
    #     # CLIP's default precision is fp16
    #     clip_model.float()

    print("Building custom CLIP")
    c_in = cfg.model.c_in
    ratio = cfg.model.ratio

    model = NewCLIP(c_in=c_in, ratio=ratio)

    return model




