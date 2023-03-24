import os
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from Lib.config import update_config, config

from Lib.utils import create_logger
from Lib.models import build_model

from torch.utils.tensorboard import SummaryWriter
from train import parse_args, build_dataloader


def main():
    args = parse_args()

    logger, final_output_dir = create_logger(
        config, args.dir_phase)

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

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

            for m in [(0, 0), (1, 1), (1, 2), (2, 2), (3, 2)]:
                z, x_title, x_norm, x_histt = net.generate_x(samples, m)
                writer.add_images(f"layer{m[0]}/raw", x_title, global_step=i)
                writer.add_images(f"layer{m[0]}/raw_norm", x_norm, global_step=i)
                writer.add_images(f"layer{m[0]}/raw_norm_hist", x_histt, global_step=i)

            # if i == num_viz:
            #     for n in range(1, m[0]+1):
            #         c_converge = eval(f"net.layer{n}.layers[{m[1]-1}].conv2.dn.c_error")
            #         closs = eval(f"net.layer{n}.layers[{m[1]-1}].conv2.dn.closs")
            #         rloss = eval(f"net.layer{n}.layers[{m[1]-1}].conv2.dn.rloss")
            #         nlayer = eval(f"net.layer{n}.layers[{m[1]-1}].conv2.dn.n_steps")
            #         step_size = eval(f"net.layer{n}.layers[{m[1]-1}].conv2.dn.step_size")
            #         lipschitz_l = 0.9 / step_size
            #
            #         converg_ratio = lipschitz_l * closs[-1] / (2*nlayer)
            #
            #         # print(f"{m}-th out loop layer, {n}-th inner loop layer")
            #         # print(c_error)
            #         writer.add_scalar(f"sample_{i}/layer{n}/lipschitz_l", lipschitz_l, global_step=1)
            #         writer.add_scalar(f"sample_{i}/layer{n}/converg_ratio", converg_ratio, global_step=1)
            #         for l in range(nlayer):
            #             writer.add_scalar(f"sample_{i}/layer{n}/c_converge", c_converge[l], global_step=l)
            #             writer.add_scalar(f"sample_{i}/layer{n}/closs", closs[l], global_step=l)
            #             writer.add_scalar(f"sample_{i}/layer{n}/rloss", rloss[l], global_step=l)

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


# CUDA_VISIBLE_DEVICES=1 python visualize.py --cfg experiments/cifar10.yaml --dir_phase cifar10_sdnet18_all_no_shortcut/viz MODEL.NAME sdnet18_all MODEL.SHORTCUT False TRAIN.MODEL_FILE logs/cifar10_sdnet18_all_no_shortcut/model_best.pth.tar
