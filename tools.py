import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    # root = "/home/peter/dxl/Code/SDNet/logs/imagenet_sdnet18_first_layer"
    root = "/Users/xilidai/dxl.cluster/SDNet/LInverse_lite/imagenet_sdnet18_viz_no_shortcut/imagenet_sdnet18_viz_niteration2"

    file = f"{root}/model_best.pth.tar"

    f = torch.load(file, map_location=torch.device('cpu'))

    writer_dict = {
        'writer': SummaryWriter(log_dir=root),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    writer = writer_dict['writer']
    print(f.keys())
    # exit()

    for k, v in f.items():

        if k == 'layer0.0.dn.weight':

            print(v.size())
            # print(v)

            writer.add_images(f"layer0.0.dn.weight", v, global_step=0)
            writer_dict['writer'].close()
            exit()





