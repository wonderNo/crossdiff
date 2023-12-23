from argparse import ArgumentParser
import os
import json
from omegaconf import OmegaConf


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--cfg",
         default='configs/crossdiff_pre.yaml', type=str)

    args_ori, extras = parser.parse_known_args()

    args = OmegaConf.load('./configs/base.yaml')
    args = OmegaConf.merge(args, OmegaConf.load(args_ori.cfg), OmegaConf.from_cli(extras))
    if 'LOCAL_RANK' in os.environ.keys():
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = -1
    args.cfg = args_ori.cfg
    args.debug = args_ori.debug

    args.save_dir = os.path.join(args.save_dir, args.NAME)
    args.ck_save_dir = os.path.join(args.save_dir, 'checkpoint')
    args.eval_dir = os.path.join(args.save_dir, 'eval')

    if args.local_rank > 0:
        args.train_platform_type = 'NoPlatform'

    if args.debug:
        args.train_platform_type = 'NoPlatform'

        args.batch_size = 4
        args.device = [0]
    
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    else:
        args.guidance_param = 2.5
    
    

    return args



