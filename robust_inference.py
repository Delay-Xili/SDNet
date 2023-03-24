import os
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Lib.config import update_config, config
from Lib.utils import create_logger
from Lib.cls_function import train, validate
from Lib.models import build_model
from train import parse_args


def gaussian_noise(images, c):
    return np.clip(images / 255 + np.random.normal(size=images.shape, scale=c), 0, 1) * 255


def shot_noise(images, c):
    return np.clip(np.random.poisson(images / 255 * c) / float(c), 0, 1) * 255


def speckle_noise(images, c):
    return np.clip(images / 255 * (1 + np.random.normal(size=images.shape, scale=c)), 0, 1) * 255


def impulse_noise(images, c):
    return np.clip(sk.util.random_noise(images / 255., mode='s&p', amount=c), 0, 1) * 255


def build_noise_transforms(noise_type="gaussian"):

    if noise_type == "gaussian":
        lambda_list = [np.arange(0.02, 0.21, 0.02), gaussian_noise]

    elif noise_type == "shot":
        lambda_list = [np.arange(60, 3, -6), shot_noise]

    elif noise_type == "speckle":
        lambda_list = [np.arange(0.1, 0.7, 0.1), speckle_noise]

    elif noise_type == "impulse":
        lambda_list = [np.arange(0.03, 0.3, 0.03), impulse_noise]

    else:
        raise NotImplementedError

    return lambda_list


def build_cifar_dataset(cfg):

    dataset_name = cfg.DATASET.DATASET
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.CIFAR10(root=f'{cfg.DATASET.ROOT}', train=False, download=True,
                                         transform=transform_valid)

    else:
        raise NotImplementedError

    return valid_dataset


def get_lmbd_acc_curve(config, data_loader, model, criterion, lmbds):

    acc_lmbd_rloss = np.zeros((len(lmbds), 3))
    for k, lmbd in enumerate(lmbds):

        model.module.layer0[0].dn.lmbd = lmbd
        result = validate(config, data_loader, model, criterion, None, None, None, None,
                          writer_dict=None)

        acc_lmbd_rloss[k, 0] = result[0]
        acc_lmbd_rloss[k, 1] = lmbd
        acc_lmbd_rloss[k, 2] = sum(model.module.layer0[0].r_loss) / len(model.module.layer0[0].r_loss)

    return acc_lmbd_rloss


def plot_lmbd_rloss_curve(final_path, noise):

    files = sorted(os.listdir(final_path))
    rloss, opt_lmbd = [], []

    for f in files:
        acc_lmbd_rloss = np.load(os.path.join(final_path, f))
        ind = np.argmax(acc_lmbd_rloss[:, 0])
        opt_lmbd.append(acc_lmbd_rloss[ind, 1])
        rloss.append(acc_lmbd_rloss[ind, 2])

    x = np.array(rloss)
    y = np.array(opt_lmbd)

    coeff = polyfit(x, y, 1)
    print(coeff)
    p = plt.plot(x, y, 'rx')
    p = plt.plot(x, coeff[0] * x + coeff[1], 'k-')
    plt.xlabel('Reconstruction Error', fontsize=20)
    plt.ylabel('$\lambda$', fontsize=20)
    plt.title('Gaussian Noise', fontsize=20)

    plt.savefig(f"{final_path}/linear_fit_{noise}_noise_test.png")


def main():
    args = parse_args()

    # dir_name = os.path.dirname(config.TRAIN.MODEL_FILE)
    logger, final_output_dir = create_logger(
        config, args.dir_phase, args.log_phase)

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # build model and load ckpt from another experiment if so.
    model = build_model(config)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    print(config)

    lambda_list = build_noise_transforms(noise_type=config.DATASET.NOISE)

    # Data loading code
    transform_valid = None  # waiting the value in the loop.
    valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True,
                                     transform=transform_valid)
    train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True,
                                     transform=transform_valid)
    gpus = list(config.GPUS)
    valid_valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    os.makedirs(os.path.join(final_output_dir, "robust_inference"), exist_ok=True)
    os.makedirs(os.path.join(final_output_dir, "robust_inference", "train_set"), exist_ok=True)
    os.makedirs(os.path.join(final_output_dir, "robust_inference", "valid_set"), exist_ok=True)

    for idx, corrpution_level in enumerate(lambda_list[0]):
        print(f"------------ noise: {config.DATASET.NOISE},  c={corrpution_level} -----------------")

        transform_valid = transforms.Compose([
            transforms.Lambda(lambd=lambda x: np.array(x)),
            transforms.Lambda(lambd=lambda x: lambda_list[1](x, corrpution_level)),
            transforms.Lambda(lambd=lambda x: x.astype(np.uint8)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        valid_valid_loader.dataset.transform = transform_valid
        valid_train_loader.dataset.transform = transform_valid

        csc_lmbds = np.linspace(0.1, 1.4, 14)

        # valid set
        acc_lmbd_rloss = get_lmbd_acc_curve(config, valid_valid_loader, model, criterion, csc_lmbds)
        np.save(f"{final_output_dir}/robust_inference/valid_set/{config.DATASET.NOISE}_{idx:03d}_c{corrpution_level}.npy", acc_lmbd_rloss)

        # train set
        # acc_lmbd_rloss_t = get_lmbd_acc_curve(config, valid_train_loader, model, criterion, csc_lmbds)
        # np.save(f"{final_output_dir}/robust_inference/train_set/{config.DATA.NOISE}_c{corrpution_level}.npy", acc_lmbd_rloss_t)

    plot_lmbd_rloss_curve(f"{final_output_dir}/robust_inference/valid_set", config.DATASET.NOISE)


if __name__ == '__main__':
    main()
