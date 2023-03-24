import torch
import torch.nn as nn

from Lib.models.resnet import ResNet18, ResNet34


def build_model(cfg):

    if cfg.MODEL.NAME == 'sdnet18':
        from Lib.models.sdnet import SDNet18
        print("SDNet18, only first layer replaced with csc layer")
        model = SDNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    elif cfg.MODEL.NAME == 'sdnet34':
        from Lib.models.sdnet import SDNet34
        print("SDNet34, only first layer replaced with csc layer")
        model = SDNet34(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    elif cfg.MODEL.NAME == 'sdnet18_all':
        from Lib.models.sdnet_inverse import SDNet18_all
        print("SDNet18_all, all layer replaced with csc layers")
        model = SDNet18_all(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'sdnet34_all':
        from Lib.models.sdnet_inverse import SDNet34_all
        print("SDNet34_all, all layer replaced with csc layers")
        model = SDNet34_all(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    elif cfg.MODEL.NAME == 'resnet18':
        model = ResNet18(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    elif cfg.MODEL.NAME == 'resnet34':
        model = ResNet34(num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    else:
        raise ValueError()

    model = nn.DataParallel(model).cuda()
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
