# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"

import json
import torch
import torch.distributed as dist

from utils.fixseed import fixseed
from utils.parser_util import parse_args
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import NoPlatform 

from diffusion import logger


def main():
    args = parse_args()
    args.running_mode = 'test'

    logger.configure(args.eval_dir, debug=args.debug, rank=args.local_rank)
    fixseed(args.seed + args.local_rank)

    train_platform = NoPlatform(args.eval_dir)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')

    logger.log("creating data loader...")
    data = None
    test_data = get_dataset_loader('humanml', batch_size=32, split='test', args=args)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    logger.log('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    logger.log("Testing init...")
    try:
        tain_loop = TrainLoop(args, train_platform, model, diffusion, data, test_data)
        logger.log(f"Start testing... ")
        if not args.test_mm:
            tain_loop.multi_eval(test_limit=20, replication_times=20, test_main=True, test_mm=False)
        else:
            tain_loop.multi_eval(test_limit=5, replication_times=5, test_main=False, test_mm=True)
    except Exception as e:
        logger.error(e)
        raise e

    logger.log('!JOB COMPLETED!')

if __name__ == "__main__":
    main()
