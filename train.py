# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import os
import json
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from utils.fixseed import fixseed
from utils.parser_util import parse_args
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

from diffusion import logger


def main():
    args = parse_args()


    logger.configure(args.save_dir, debug=args.debug, rank=args.local_rank)

    fixseed(args.seed + args.local_rank)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    args_path = os.path.join(args.save_dir, 'config.yaml')

    if not os.path.exists(args_path) and args.local_rank < 1 and not args.debug:
        OmegaConf.save(config=args, f=args_path)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')


    logger.log("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, args=args)
    test_data = get_dataset_loader('humanml', batch_size=32, split='test',  args=args)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    

    logger.log('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    logger.log("Training init...")
    try:
        tain_loop = TrainLoop(args, train_platform, model, diffusion, data, test_data)
        logger.log("Start training...")
        tain_loop.run_loop()
    except Exception as e:
        logger.error(e)
        raise e
    train_platform.close()

    logger.log('!JOB COMPLETED!')

if __name__ == "__main__":
    main()
