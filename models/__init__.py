from . import Minkowski, ppat, pointnet2

def make(config):
    if config.model.name == "MinkowskiFCNN":
        model = Minkowski.MinkowskiFCNN(config)
    elif config.model.name == "MinkResNet":
        model = Minkowski.MinkResNet(config)
    elif config.model.name == "MinkResNet34":
        model = Minkowski.MinkResNet34(config)
    elif config.model.name == "MinkResNet34_Two":
        model = Minkowski.MinkResNet34_Two(config)
    elif config.model.name == "MinkResNet34_2adapter":
        model = Minkowski.MinkResNet34_2adapter(config)
    elif config.model.name == "MinkResNet11":
        model = Minkowski.MinkResNet11(config)
    elif config.model.name == "MinkowskiFCNN_small":
        model = Minkowski.MinkowskiFCNN_small(config)
    elif config.model.name == "PointBERT":
        model = ppat.make(config)
    elif config.model.name == "DGCNN":
        from . import dgcnn
        model = dgcnn.make(config)
    elif config.model.name == "PointNeXt":
        from . import pointnext
        model = pointnext.make(config)
    elif config.model.name == "PointMLP":
        from . import pointmlp_ulip
        model = pointmlp_ulip.make(config)
    elif config.model.name == "PointNet":
        from . import pointnet
        model = pointnet.make(config)
    elif config.model.name == "PointNet2":
        from . import pointnet
        model = pointnet2.make(config)
    elif config.model.name == "CustomCLIP":
        from . import prompt_pretrainer
        model = prompt_pretrainer.make(config)
    elif config.model.name == "CLIPAdapter":
        from . import clip_adapter
        model = clip_adapter.make(config)
    elif config.model.name == "ULIP_PointBert":
        from .ulip import ULIP_models
        model = ULIP_models.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
