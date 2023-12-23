from typing import Any
import torch
from torch.utils import data
import numpy as np
import os
import json
import random

from diffusion import logger

class UCF101(data.Dataset):
    def __init__(self, args):
        
        self.dataset_name = 'ucf101'
        self.max_motion_length = 196
        self.use_mean_joint = args.use_mean_joint
        self.joint_mask_ratio = args.joint_mask_ratio
        self.mask_joint = self.joint_mask_ratio > 0

        joints_dir = os.path.join(args.ucf_root, 'complicate_2d')
        text_json = os.path.join(args.ucf_root, 'text.json')
        self.mean_joint = np.load(os.path.join(args.data_root, 'Mean_complicate2d.npy'))
        self.std_joint = np.load(os.path.join(args.data_root, 'Std_complicate2d.npy'))

        with open(text_json) as f:
            self.text_dict = json.load(f)

        file_list = os.listdir(joints_dir)

        self.data = []
        if not args.ucf_keys:
            select = False
        else:
            select = True

        # count = 0
        for file in file_list:
            tag = file.split('_')[1]
            if (select and tag not in args.ucf_keys) or tag not in self.text_dict.keys():
                continue
            
            joint = np.load(os.path.join(joints_dir, file)).astype(np.float32)
            if len(joint) < 40 or len(joint) > 196:
                continue

            # mask[:,[0,1,86,87]] = 0
            if joint.shape[-1] == 134:
                mask = np.ones_like(joint)
            else:
                mask = joint[:,134:]
                joint = joint[:,:134]
            self.data.append({'joint': joint,
                              'tag': tag,
                              'm_length': len(joint),
                              'mask':mask
                              })
            # count += 1
            # if count > 5:
            #     break
            
        logger.info(f'UCF101 dataset has {self.__len__()} samples..')

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        joint_dict = self.data[index]
        joint, tag, m_length = joint_dict['joint'], joint_dict['tag'], joint_dict['m_length']
        joint_mask = joint_dict['mask']
        text = random.choice(self.text_dict[tag])

        # coin2 = np.random.choice(['single', 'single', 'double'])

        # if coin2 == 'double':
        #     m_length = (m_length // 4 - 1) * 4
        # elif coin2 == 'single':
        #     m_length = (m_length // 4) * 4
        # idx = random.randint(0, len(joint) - m_length)
        # joint = joint[idx:idx + m_length]
        # joint_mask = joint_mask[idx:idx + m_length]

        "Z Normalization"
        # joint_mask = np.ones_like(joint)
        
        if self.use_mean_joint:
            joint = (joint - self.mean_joint) / self.std_joint

        motion_mask = np.ones((m_length, 263), dtype=np.float32)

        if m_length < self.max_motion_length:
            joint = np.concatenate([joint,
                                     np.zeros((self.max_motion_length - m_length, joint.shape[1]), dtype=np.float32)
                                     ], axis=0)
            joint_mask = np.concatenate([joint_mask,
                                     np.zeros((self.max_motion_length - m_length, joint_mask.shape[1]), dtype=np.float32)
                                     ], axis=0)
            motion_mask = np.concatenate([motion_mask,
                                     np.zeros((self.max_motion_length - m_length, motion_mask.shape[1]), dtype=np.float32)
                                     ], axis=0)
        motion = np.zeros((self.max_motion_length, 263), dtype=np.float32)
        
        return (text, motion, m_length, joint, False, joint_mask, motion_mask)