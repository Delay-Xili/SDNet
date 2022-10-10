import os
import shutil
import argparse
import pprint
import numpy as np

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
from Lib.datasets.robust import CIFAR10_C, ImageNet_C_dataloader_generator, CIFAR10_erase_offline
from Lib.datasets.robust import corruption_imagenet as CORRUPTIONS
from Lib.models import build_model
from train import parse_args


# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def build_dataset(cfg):
    dataset_name = cfg.DATASET.DATASET
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.CIFAR10(root=f'{cfg.DATASET.ROOT}', train=False, download=True,
                                         transform=transform_valid)
    elif dataset_name == 'cifar10-c':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        valid_dataset = CIFAR10_C(root=f'{cfg.DATASET.ROOT}', transform=transform_valid, level=cfg.DATASET.NOISE_LEVEL)
    elif dataset_name in ['cifar10_occlusion']:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = CIFAR10_erase_offline(root=f'{cfg.DATASET.ROOT}', transform=transform_valid)

    else:
        raise NotImplementedError

    return valid_dataset


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce


def main():
    args = parse_args()

    dir_name = os.path.dirname(config.TRAIN.MODEL_FILE)
    logger, final_output_dir = create_logger(
        config, dir_name, args.log_phase)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # build model and load ckpt from another experiment if so.
    model = build_model(config)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    print(config)

    if config.DATASET.DATASET in ["imagenet_c"]:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])

        valid_loader = ImageNet_C_dataloader_generator(config.DATASET.ROOT, transform_valid, config.TEST.BATCH_SIZE_PER_GPU)

    else:
        # Data loading code
        valid_dataset = build_dataset(config)
        gpus = list(config.GPUS)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

    dataset_name = config.DATASET.DATASET
    if dataset_name == 'cifar10-c':
        overall = []
        names = []
        while valid_dataset.data is not None:
            logger.info(f"------------ {valid_dataset.data_name} -----------------")
            results = validate(config, valid_loader, model, criterion, None, None, output_dir=final_output_dir,
                     tb_log_dir=None, writer_dict=None)
            overall.append(results[0].cpu().numpy())
            names.append(valid_dataset.data_name)
            valid_dataset.next_dataset()
            logger.info(f"------------ ------------------------- -----------------")
        logger.info(f"overall: {sum(overall) / len(overall)}")

        csv_file = '{}/{}.csv'.format(final_output_dir, args.log_phase)
        with open(os.path.join(csv_file), 'a') as f:
            f.write('head \t means \n')
            for k in range(len(overall)):
                f.write('\t'.join([names[k]] + [str(overall[k])] + ['\n']))
            f.write('\t'.join(["overall"] + [str(sum(overall) / len(overall))] + ['\n']))

    elif dataset_name == 'cifar10':
        logger.info(f"--------------check bug---------------")
        validate(config, valid_loader, model, criterion, None, None, output_dir=final_output_dir,
                 tb_log_dir=None, writer_dict=None)
        logger.info(f"------------ ------------------------- -----------------")

    elif dataset_name in ['cifar10_occlusion']:
        while valid_dataset.data is not None:
            logger.info(f"------------ {valid_dataset.data_list[valid_dataset.dataset_id - 1]} -----------------")
            validate(config, valid_loader, model, criterion, None, None, output_dir=final_output_dir,
                     tb_log_dir=None, writer_dict=None)
            valid_dataset.next_dataset()
            logger.info(f"------------ ------------------------- -----------------")

    elif dataset_name in ['imagenet_c']:
        corruption_accs = {}
        for (c, num, val) in valid_loader:
            logger.info(f"------------ {c}: {num} -----------------")
            results = validate(config, val, model, criterion, None, None, output_dir=final_output_dir,
                               tb_log_dir=None, writer_dict=None)
            # results = (0.1, 2.0)

            if c in corruption_accs:
                corruption_accs[c].append(results[0].cpu().numpy())
            else:
                corruption_accs[c] = [results[0].cpu().numpy()]

        csv_file = '{}/{}.csv'.format(final_output_dir, args.log_phase)
        with open(os.path.join(csv_file), 'a') as f:
            f.write('head \t level1 \t level2 \t level3 \t level4 \t level5\n')
            for c in CORRUPTIONS:
                f.write('\t'.join([c] + list(map(str, corruption_accs[c])) + ['\n']))

        for c in CORRUPTIONS:
            logger.info('\t'.join([c] + list(map(str, corruption_accs[c]))))

        logger.info(f'mCE (normalized by AlexNet): {compute_mce(corruption_accs)}')

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
