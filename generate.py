import os
import torch
import numpy as np

from utils.parser_util import parse_args
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.load_utils import encode_text, find_resume_checkpoint, load_and_freeze_clip
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_2d_motion
from diffusion import logger


def main():
    args = parse_args()
    args.debug = True

    logger.configure(args.save_dir, debug=True)

    device = torch.device('cuda')

    dataset = get_dataset_loader('humanml', split='generate', args=args)
    
    model, diffusion = create_model_and_diffusion(args)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.test_checkpoint:
        resume_checkpoint = args.test_checkpoint
    else:
        resume_checkpoint, last_epoch = find_resume_checkpoint(args.ck_save_dir)
    print(f"loading model from checkpoint: {resume_checkpoint}...")
    model.load_state_dict(torch.load(resume_checkpoint, map_location='cpu'))

    

    clip_model = load_and_freeze_clip()
    clip_model.cuda()

    os.makedirs(args.eval_dir, exist_ok=True)
    captions = list(args.captions) * args.sample_times


    args.nsamples = len(captions)
    
    model_kwargs = {}
    model_kwargs['enc_text'] = encode_text(clip_model, captions, device)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model) 
    model.cuda()
    model.eval()


    skeleton = paramUtil.t2m_kinematic_chain
    fps = 20
    os.makedirs(args.eval_dir, exist_ok=True)

    # directly generate 3D motion
    if args.generate_3d:
        sample = diffusion.p_sample_loop(
            model,
            (args.nsamples, 263, 1, 196),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample = sample[:,:,0].permute(0,2,1)
        sample = dataset.inv_transform(sample)
        sample = recover_from_ric(sample, 22).cpu().numpy()

    
        for i in range(args.nsamples):
            print(f'generating {i}')
            animation_save_path = os.path.join(args.eval_dir, f'{i}.mp4')
            motion = sample[i]
            caption = captions[i]
            np.save(os.path.join(args.eval_dir, f'{i}.npy'), motion)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
    


    # sample first in 2D domain and then in 3D domain
    if args.generate_2d:
        sample = diffusion.p_sample_loop(
            model,
            (args.nsamples, 134, 1, 196),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        t = torch.tensor([0]*args.nsamples, device=device)
        joint_predict = model.model(sample, t, model_kwargs['enc_text'], return_m=True, return_j=True, force_mask=True)

        sample = joint_predict['m'].detach()
        sample = sample[:,:,0].permute(0,2,1)
        sample = dataset.inv_transform(sample)
        sample = recover_from_ric(sample, 22).cpu().numpy()

    
        for i in range(args.nsamples):
            print(f'generating {i}')
            animation_save_path = os.path.join(args.eval_dir, f'{i}_3dfrom2d.mp4')
            motion = sample[i]
            caption = captions[i]
            np.save(os.path.join(args.eval_dir, f'{i}_3dfrom2d.npy'), motion)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
    



if __name__ == "__main__":
    main()
