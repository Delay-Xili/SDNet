import torch
import torch.nn as nn

import yaml
import os
import logging
import time
import pprint
from pathlib import Path


def create_logger(cfg, dir_phrase='', log_phrase='train'):
    root_output_dir = Path(cfg.LOG_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    # model = cfg.MODEL.NAME
    # ------ output pth name -----------
    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    # final_output_dir = root_output_dir / (cfg_name + '_' + dir_phrase)
    final_output_dir = root_output_dir / dir_phrase
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # ------ log file name -----------
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, log_phrase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    print('=> creating {}'.format(final_log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(pprint.pformat(cfg))
    logger.info(pprint.pformat(str(final_output_dir)))

    return logger, str(final_output_dir)


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def model_summary(model):
    params_sum = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    return f"params_sum: {params_sum / 1000000.0} M"


def _to_yaml(obj, filename=None, default_flow_style=False,
             encoding="utf-8", errors="strict",
             **yaml_kwargs):
    if filename:
        with open(filename, 'w',
                  encoding=encoding, errors=errors) as f:
            yaml.dump(obj, stream=f,
                      default_flow_style=default_flow_style,
                      **yaml_kwargs)
    else:
        return yaml.dump(obj,
                         default_flow_style=default_flow_style,
                         **yaml_kwargs)