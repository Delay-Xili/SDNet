import os
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Lib.config import update_config, config

from Lib.utils import create_logger, save_checkpoint, model_summary
from Lib.models import build_model

from torch.utils.tensorboard import SummaryWriter
from train import parse_args


def build_dataloader(cfg):
    gpus = list(cfg.GPUS)
    dataset_name = cfg.DATASET.DATASET
    # dataset_name = 'imagenet'
    if 'cifar' in dataset_name:

        if dataset_name == 'cifar10':
            dataset = datasets.CIFAR10
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100
        else:
            raise ValueError

        if config.VIZ_INPUTNORM:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),  #
            # transforms.RandomHorizontalFlip(),
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

        # traindir = '/home/dxl/Data/ILSVRC2012/train'
        # valdir = '/home/dxl/Data/ILSVRC2012/val'
        # IMAGE_SIZE = (224, 224)

        traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
        IMAGE_SIZE = config.MODEL.IMAGE_SIZE

        if config.VIZ_INPUTNORM:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE[0]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(IMAGE_SIZE[0]),
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
        shuffle=False,
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


def main():
    args = parse_args()

    logger, final_output_dir = create_logger(
        config, args.dir_phase)

    # TODO copy the training log to the viz target path. (maybe have some potential bugs)
    p = os.path.dirname(config.TRAIN.MODEL_FILE)
    os.system(f"cp {p}/*.log {final_output_dir}")

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # build model and load ckpt from another experiment if so.
    model = build_model(config)

    # setting tensorboard writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=final_output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    writer = writer_dict['writer']
    # global_steps = writer_dict['train_global_steps']

    train_loader, valid_loader = build_dataloader(config)

    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    net.eval()

    train = config.VIZ_TRAINSET
    load_dataset = train_loader if train else valid_loader

    num_viz = 5
    with torch.no_grad():
        for i, (input, target) in enumerate(load_dataset):

            net.update_stepsize()  # bug here, makes the visualization of last several layer bad.
            # output, extra = net(input)

            print("i: ", i)
            num = 80 if 'cifar' in config.DATASET.DATASET else 8

            samples = input.cuda()[:num]
            _, indices = torch.sort(target[:num])
            samples = samples[indices]

            if config['VIZ_INPUTNORM']:
                if 'cifar' in config['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.2023, 0.1994, 0.2010])[None, :, None, None].to('cuda')
                elif 'imagenet' in config['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
                else:
                    raise ValueError()
                writer.add_images("input", samples * std + mean, global_step=i)
            else:
                writer.add_images("input", samples, global_step=i)

            for m in [1, 2, 3, 4, 5, 6, 7, 8]:
                z, x_title, x_norm, x_histt = net.generate_x(samples, m)
                writer.add_images(f"layer{m}/raw", x_title, global_step=i)
                writer.add_images(f"layer{m}/raw_norm", x_norm, global_step=i)
                writer.add_images(f"layer{m}/raw_norm_hist", x_histt, global_step=i)

            if i == num_viz:
                for n in range(1, m+1):
                    c_converge = eval(f"net.layer{n}.dn.c_error")
                    closs = eval(f"net.layer{n}.dn.closs")
                    rloss = eval(f"net.layer{n}.dn.rloss")
                    nlayer = eval(f"net.layer{n}.dn.n_steps")
                    step_size = eval(f"net.layer{n}.dn.step_size[0]")
                    lipschitz_l = 0.9 / step_size

                    converg_ratio = lipschitz_l * closs[-1] / (2*nlayer)

                    # print(f"{m}-th out loop layer, {n}-th inner loop layer")
                    # print(c_error)
                    writer.add_scalar(f"sample_{i}/layer{n}/lipschitz_l", lipschitz_l, global_step=1)
                    writer.add_scalar(f"sample_{i}/layer{n}/converg_ratio", converg_ratio, global_step=1)
                    for l in range(nlayer):
                        writer.add_scalar(f"sample_{i}/layer{n}/c_converge", c_converge[l], global_step=l)
                        writer.add_scalar(f"sample_{i}/layer{n}/closs", closs[l], global_step=l)
                        writer.add_scalar(f"sample_{i}/layer{n}/rloss", rloss[l], global_step=l)

            # z0, x_title0, x_norm0, x_hist0 = net.generate_x(samples, (0, 1))
            # writer.add_images("layer0/raw", x_title0, global_step=i)
            # writer.add_images("layer0/raw_norm", x_norm0, global_step=i)
            # writer.add_images("layer0/raw_norm_hist", x_hist0, global_step=i)
            #
            # for m in [1, 2, 3, 4]:
            #     for n in [1, 2]:
            #         z, x_title, x_norm, x_histt = net.generate_x(samples, (m, n))
            #         writer.add_images(f"module{m}/block{n}/raw", x_title, global_step=i)
            #         writer.add_images(f"module{m}/block{n}/raw_norm", x_norm, global_step=i)
            #         writer.add_images(f"module{m}/block{n}/raw_norm_hist", x_histt, global_step=i)

            torch.cuda.empty_cache()
            writer_dict['writer'].flush()
            if i > num_viz:
                break

        writer_dict['writer'].close()


if __name__ == '__main__':
    main()
