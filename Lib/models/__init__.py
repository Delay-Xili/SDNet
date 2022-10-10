import torch
import torch.nn as nn

# from Lib.models.stack_dictnet import StackDictnet
# from Lib.models.sdnet import SDNet18, SDNet50
# from Lib.models.sdnet_inverse import SDNet18, SDNet50
from Lib.models.resnet import ResNet18, ResNet34
from Lib.models.osdnet import OSDNet18


def build_model(cfg):

    if cfg.MODEL.NAME == 'sdnet18':
        from Lib.models.sdnet import SDNet18
        model = SDNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'sdnet18_only_first_layer':
        from Lib.models.sdnet import SDNet18_sdlayer_only_for_first_layer
        print("SDNet18_sdlayer_only_for_first_layer")
        model = SDNet18_sdlayer_only_for_first_layer(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    elif cfg.MODEL.NAME == 'sdnet34':
        from Lib.models.sdnet import SDNet34
        model = SDNet34(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'sdnet18_viz':
        from Lib.models.sdnet_inverse import SDNet18
        model = SDNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'sdnet34_viz':
        from Lib.models.sdnet_inverse import SDNet34
        model = SDNet34(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif 'lenet' in cfg.MODEL.NAME:
        from Lib.models.lenet import LeNet
        model = LeNet(num_classes=cfg.MODEL.NUM_CLASSES)

    elif cfg.MODEL.NAME == 'osdnet18':
        model = OSDNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    elif cfg.MODEL.NAME == 'resnet18':
        model = ResNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'resnet34':
        model = ResNet34(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    else:
        raise ValueError()

    gpus = list(cfg.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    with torch.no_grad():
        print("====================")
        inputx = torch.zeros([32, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[0]]).cuda()
        # print(inputx)
        _ = model.module(inputx)
        print("====================")

    if cfg.TRAIN.MODEL_FILE:
        # load ckpt from any other experiment result.
        model.module.load_state_dict(torch.load(cfg.TRAIN.MODEL_FILE))
        # model.load_state_dict(torch.load(cfg.TRAIN.MODEL_FILE))
        print('=> loading model from {}'.format(cfg.TRAIN.MODEL_FILE))
        # logger.info('=> loading model from {}'.format(cfg.TRAIN.MODEL_FILE))


    return model
