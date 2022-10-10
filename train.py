import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Lib.config import update_config, config

from Lib.utils import create_logger, save_checkpoint, model_summary, _to_yaml
from Lib.cls_function import train, validate
from Lib.models import build_model
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
randomSeed = 1
random.seed(randomSeed)  # python random seed
torch.manual_seed(randomSeed)  # pytorch random seed
np.random.seed(randomSeed)  # numpy random seed



def get_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), # use the parameters with requires_grad=True
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    else:
        raise ValueError()

    return optimizer


def build_dataloader(cfg): # support cifar10 and cifar100 and imagenet
    gpus = list(cfg.GPUS)
    dataset_name = cfg.DATASET.DATASET
    if 'cifar' in dataset_name:

        if dataset_name == 'cifar10': 
            dataset = datasets.CIFAR10
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100
        else:
            raise ValueError

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = dataset(root=f'{cfg.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = dataset(root=f'{cfg.DATASET.ROOT}', train=False, download=True, transform=transform_valid)

    elif dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    return train_loader, valid_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--cfg',
                        help='the default setting is sdnet18 for cifar10 if no specific .yaml be chosen',
                        type=str,
                        )
    parser.add_argument('--dir_phase',
                        help='the name for each experiment',
                        type=str,
                        default='train')
    parser.add_argument('--log_phase',
                        help='the name for each validation (specific for robust test)',
                        type=str,
                        default='valid')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir = create_logger(
        config, args.dir_phase)

    _to_yaml(config, os.path.join(final_output_dir, 'config.yaml'))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # build model and load ckpt from another experiment if so.
    model = build_model(config)
    logger.info("model summary")
    logger.info(model_summary(model))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    # setting tensorboard writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=final_output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf = 0.0
    best_loss_total, best_loss_cls, best_loss_r, best_loss_c = 0.0, 0.0, 0.0, 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        # resume ckpt from current pth when the experiment was interrupted for some reasons.
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])

            # Update weight decay if needed
            checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
            optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint:
                if config.TRAIN.LR_SCHEDULER != 'step':
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 1e5, last_epoch=checkpoint['lr_scheduler']['last_epoch'])
                elif isinstance(config.TRAIN.LR_STEP, list):
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                        last_epoch - 1)
                else:
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                        last_epoch - 1)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            best_model = True

    # Data loading code
    dataset_name = config.DATASET.DATASET
    train_loader, valid_loader = build_dataloader(config)

    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader) * config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch - 1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch - 1)
        logger.info(f"lr_scheduler: {config.TRAIN.LR_SCHEDULER}")

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        topk = (1,) if dataset_name == 'cifar10' else (1, 5)

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, None, writer_dict, topk=topk)
        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()

        torch.cuda.empty_cache()
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion, lr_scheduler, epoch,
                                  final_output_dir, None, writer_dict, topk=topk)
        torch.cuda.empty_cache()
        writer_dict['writer'].flush()

        if perf_indicator[0] > best_perf:
            best_perf = perf_indicator[0]
            best_loss_total = perf_indicator[1]
            best_loss_cls = perf_indicator[2]
            best_loss_r = perf_indicator[3]
            best_loss_c = perf_indicator[4]
            best_model = True
        else:
            best_model = False
        logger.info('Test: Best Accuracy@1 {:.4f}, total_loss {:.4f}, cls_loss {:.4f}, r_loss {:.4f}, c_loss {:.4f}'.format(best_perf, best_loss_total, best_loss_cls, best_loss_r, best_loss_c))
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            # 'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator[0],
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
