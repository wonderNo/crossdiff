import os
import time
import numpy as np
import copy
import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from utils.fixseed import fixseed
from diffusion import logger
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorWrapper
from eval.eval_humanml import evaluation_transformer, evaluation_transformer_mm
from utils.load_utils import find_resume_checkpoint, \
                    log_loss_dict, load_and_freeze_clip, encode_text, masked_l2
from utils.filter import OneEuroFilter


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data, test_data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.weight_decay = args.weight_decay

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_epochs = args.num_epochs

        self.mode = args.mode

        if self.mode in ['finetune_ucf']:
            self.filter = OneEuroFilter()
        


        if args.local_rank != -1:
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.device = torch.device(f"cuda:0")


        self.save_dir = args.save_dir
        self.debug = args.debug
        self.ck_save_dir = args.ck_save_dir
        self.start_epoch = 0
        self.epoch = 0
        if args.local_rank < 1:
            os.makedirs(self.ck_save_dir, exist_ok=True)

        self._load_and_sync_parameters()

        self.model.to(self.device)
        if self.mode in ['finetune']:
            find_unuesd = True
        else:
            find_unuesd = False
        if args.local_rank != -1:
            self.model = DistributedDataParallel(self.model, 
                                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=find_unuesd)


        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.clip_model = load_and_freeze_clip()
        self.clip_model.to(self.device)

        self.eval_wrapper = EvaluatorWrapper(self.device, args.eval_part)

        self.test_data = test_data

    def _load_and_sync_parameters(self):
        # for test
        if self.args.running_mode == 'test' and self.args.test_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.args.test_checkpoint, map_location='cpu'))
            logger.log(f"loading model from checkpoint: {self.args.test_checkpoint}...")
            return 0
        
        

        resume_checkpoint, last_epoch = find_resume_checkpoint(self.ck_save_dir)

        if resume_checkpoint:
            self.start_epoch = last_epoch + 1
            self.epoch = self.start_epoch
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(torch.load(resume_checkpoint, map_location='cpu'))
            fixseed(self.args.seed + self.args.local_rank + self.start_epoch)
        elif self.args.resume_checkpoint:
            logger.log(f"loading model from resume_checkpoint: {self.args.resume_checkpoint}...")
            self.model.load_state_dict(torch.load(self.args.resume_checkpoint, map_location='cpu'))
            fixseed(self.args.seed + self.args.local_rank + 10)
        else:
            logger.log("start first training...")

        if self.mode in ['finetune_ucf']:
            self.teacher = copy.deepcopy(self.model)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.to(self.device)


        return last_epoch


    def run_loop(self):
        training_keys = ['motion', 'joint', 'valid', 'mask', 'joint_mask', 'motion_mask']

        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            if self.args.local_rank != -1:
                self.data.sampler.set_epoch(epoch)

            if self.args.local_rank < 1:
                bar =tqdm(self.data)
            else:
                bar = self.data

            for batch in bar:
                cuda_batch = {}
                for k, v in batch.items():
                    if k == 'text':
                        cuda_batch['enc_text'] = encode_text(self.clip_model, v, self.device)
                    elif k in training_keys:
                        cuda_batch[k] = v.to(self.device)
                cuda_batch['length'] = max(batch['lengths'] * (~ batch['valid']))

                self.forward_backward(cuda_batch)

            if self.epoch % self.args.log_interval == 0:
                for k,v in logger.get_current().name2val.items():
                    if k == 'loss':
                        logger.log('epoch[{}]: loss[{:0.5f}]'.format(self.epoch, v))

                    if k in ['step', 'samples'] or '_q' in k:
                        continue
                    else:
                        self.train_platform.report_scalar(name=k, value=v, iteration=self.epoch, group_name='Loss')

            if self.epoch % self.args.eval_interval == 0 and self.args.eval_during_train:
                self.evaluate(test_limit=5)

            if self.epoch % self.args.save_interval == 0:
                if self.args.local_rank < 1 and not self.args.debug:
                    self.save()
                logger.log(f'save in epoch:{self.epoch}.')


        # Save the last checkpoint if it wasn't already saved.
        if self.epoch % self.save_interval != 0:
            if self.args.local_rank < 1 and not self.args.debug:
                self.save()
            if self.args.eval_during_train:
                self.multi_eval()
                
    def multi_eval(self, test_limit=10, replication_times=5, test_mm=False, test_main=True):
        results = {}
        for i in range(replication_times):
            if self.args.local_rank != -1:
                self.test_data.sampler.set_epoch(i)
            res = self.evaluate(test_limit=test_limit, test_mm=test_mm, test_main=test_main)

            for key, item in res.items():
                if key not in results:
                    results[key] = [item]
                else:
                    results[key] += [item]

        log_info = f'multi_eval for test_limit {test_limit} and replication times {replication_times}\n'
        for k,v in results.items():

            mean = np.mean(v, axis=0)
            std = np.std(v, axis=0)
            conf_interval = 1.96 * std / np.sqrt(replication_times)
            log_info = log_info + f'{k}: mean:{mean} conf:{conf_interval}\n'

        logger.info(log_info)


    def evaluate(self, test_limit=10000, test_mm=False, test_main=True):
        self.model.eval()
        start_eval = time.time()

        res = {}

        if test_main:
            res = evaluation_transformer(self.args,
                                val_loader=self.test_data,
                                model = self.model,
                                diffusion=self.diffusion,
                                clip_model=self.clip_model,
                                eval_wrapper=self.eval_wrapper,
                                device=self.device,
                                nb_iter=self.epoch,
                                test_limit=test_limit)
        if test_mm:
            res2 = evaluation_transformer_mm(self.args,
                               val_loader=self.test_data,
                               model = self.model,
                               diffusion=self.diffusion,
                               clip_model=self.clip_model,
                               eval_wrapper=self.eval_wrapper,
                               device=self.device,
                               nb_iter=self.epoch,
                               test_limit=test_limit)
            res.update(res2)

        end_eval = time.time()
        # self.model.train()
        logger.log(f'Evaluation time: {round(end_eval-start_eval)/60:.2f}min')
        return res



    def forward_backward(self, batch):
        self.model.zero_grad()
        

        losses = {}

        if self.mode in ['finetune']:
            t, weights = self.schedule_sampler.sample(batch['motion'].shape[0], self.device)

            motion = batch['motion']
            noise = torch.randn_like(motion)
            motion_t = self.diffusion.q_sample(motion, t, noise=noise)

            mask3d = batch['mask']

            motion_predict = self.model(motion_t, t, batch['enc_text'], return_m=True, return_j=False) 
            
            losses["loss"] = masked_l2(motion, motion_predict['m'], mask3d)

            loss = losses['loss'].mean()
            log_loss_dict(losses)
            loss.backward()
            self.opt.step()


        elif self.mode in ['pretrain']:
            t, weights = self.schedule_sampler.sample(batch['motion'].shape[0], self.device)

            motion = batch['motion']
            joint = batch['joint']
            noise = torch.randn_like(motion)
            motion_t = self.diffusion.q_sample(motion, t, noise=noise)

            mask3d = batch['valid'].unsqueeze(1).unsqueeze(1).unsqueeze(1) * batch['mask']
            mask2d = batch['joint_mask']
            mask3d2d = batch['valid'].unsqueeze(1).unsqueeze(1).unsqueeze(1) * batch['joint_mask']

            motion_predict = self.model(motion_t, t, batch['enc_text'], return_m=True, return_j=True)
            
            noise_joint = torch.randn_like(joint)
            joint_t = self.diffusion.q_sample(joint, t, noise=noise_joint)
            joint_predict = self.model(joint_t, t, batch['enc_text'], return_m=True, return_j=True)

            losses["motion"] = masked_l2(motion, motion_predict['m'], mask3d)
            losses["motion2joint"] = masked_l2(joint, motion_predict['j'], mask3d2d)
            losses["joint"] = masked_l2(joint, joint_predict['j'], mask2d)
            losses["joint2motion"] = masked_l2(motion, joint_predict['m'], mask3d)

            losses['loss'] = losses["motion"] + losses["joint"] + \
                self.args.w_m2j * losses["motion2joint"] + self.args.w_j2m * losses["joint2motion"]

            loss = losses['loss'].mean()
            log_loss_dict(losses)
            loss.backward()
            self.opt.step()


        elif self.mode in ['finetune_ucf']:
            t = torch.tensor([0] * batch['motion'].shape[0], device=self.device)
            with torch.no_grad():
                joint_predict = self.teacher(batch['joint'] * batch['joint_mask'], t, batch['enc_text'], return_m=True, return_j=False, force_mask=True)
                motion_0_j = joint_predict['m']
                
                for j in range(batch['length']):
                    motion_0_j[:,:,:,j] = self.filter.filter_signal(motion_0_j[:,:,:,j])


            valid = batch['valid'].unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            motion = batch['motion'] * valid + motion_0_j * (1 - valid)
            joint = batch['joint']
 

            t, weights = self.schedule_sampler.sample(batch['motion'].shape[0], self.device)

            mask = batch['mask'] * valid + batch['motion_mask'] * (1 - valid)

            noise = torch.randn_like(motion)
            motion_t = self.diffusion.q_sample(motion * batch['motion_mask'], t, noise=noise)
            motion_predict = self.model(motion_t, t, batch['enc_text'], return_m=True, return_j=True)
            
            noise_joint = torch.randn_like(joint)
            joint_t = self.diffusion.q_sample(joint * batch['joint_mask'], t, noise=noise_joint)
            joint_predict = self.model(joint_t, t, batch['enc_text'], return_m=True, return_j=True)

            losses["motion"] = masked_l2(motion, motion_predict['m'], mask)
            losses["motion2joint"] = masked_l2(joint, motion_predict['j'], batch['joint_mask'])
            losses["joint"] = masked_l2(joint, joint_predict['j'], batch['joint_mask'])
            losses["joint2motion"] = masked_l2(motion, joint_predict['m'], mask)

            losses['loss'] = losses["motion"] + losses["joint"] + \
                self.args.w_m2j * losses["motion2joint"] + self.args.w_j2m * losses["joint2motion"]

            loss = losses['loss'].mean()
            log_loss_dict(losses)
            loss.backward()
            self.opt.step()

        else:
            raise NotImplementedError(f'not implement {self.mode}')

    
    def save(self):
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        filename = f"model{(self.epoch):04d}.pt"
        torch.save(state_dict, os.path.join(self.ck_save_dir, filename))
