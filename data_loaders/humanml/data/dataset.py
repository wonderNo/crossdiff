import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
# import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt

from diffusion import logger
# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    if len(batch[0]) > 7:
        batch = [b[:7] for b in batch]
    return default_collate(batch)


class Text2MotionJointDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mean_joint, std_joint, split='train', args=None):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        self.joint_mask_ratio = args.joint_mask_ratio
        self.mask_joint = self.joint_mask_ratio > 0

        self.relate_feats3d = [[0,1,2,3,193,194]] +\
    [[4+i*3,5+i*3,6+i*3,67+i*6,68+i*6,69+i*6,70+i*6,71+i*6,72+i*6,195+i*3,196+i*3,197+i*3] for i in range(21)]
        self.relate_feats3d[7].append(259)
        self.relate_feats3d[10].append(260)
        self.relate_feats3d[8].append(261)
        self.relate_feats3d[11].append(262)
        self.relate_feats2d = [[0,1,86,87]] +\
    [[2+i*2,3+i*2,44+i*2,45+i*2,88+i*2,89+i*2] for i in range(21)]
        self.relate_feats2d[7].append(130)
        self.relate_feats2d[10].append(131)
        self.relate_feats2d[8].append(132)
        self.relate_feats2d[11].append(133)

        motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        text_dir = pjoin(opt.data_root, 'texts')
        joints_dir = pjoin(opt.data_root, 'new_joints2d_complicate')


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        

        id_list2 = []
        for ids in id_list:
            if ids in id_list2:
                continue
            if ids[0] == 'M':
                Mid = ids[1:]
            else:
                Mid = 'M' + ids
            id_list2.append(ids)
            id_list2.append(Mid)
        assert len(id_list2) == len(id_list)
        id_list = id_list2
        motion_ratio = 1 if split != 'train' else args.motion_ratio
        len_data = 200 if args.debug else len(id_list)
        cutting_id = int(len_data * motion_ratio)

        if args.cut_2d:
            id_list = id_list[:cutting_id]

        if args.debug:
            id_list = id_list[:64]

        if args.running_mode == 'test' and split == 'train':
            id_list = id_list[:32]

        new_name_list = []
        length_list = []

        if args.local_rank < 1:
            bar = enumerate(tqdm(id_list))
        else:
            bar = enumerate(id_list)

        for i, name in bar:
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                joint = []
                joint.append(np.load(pjoin(joints_dir, name + "-0.npy")))
                joint.append(np.load(pjoin(joints_dir, name + "-1.npy")))
                joint.append(np.load(pjoin(joints_dir, name + "-2.npy")))
                joint.append(np.load(pjoin(joints_dir, name + "-3.npy")))
                if i >= cutting_id:
                    valid = False
                else:
                    valid = True
                motion = motion[:len(joint[0])]
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                n_joint = [j[int(f_tag * 20):int(to_tag * 20)] for j in joint]
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                       "joint": n_joint,
                                                        "valid": valid}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                       "joint": joint,
                                        "valid": valid}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.mean_joint = mean_joint
        self.std_joint = std_joint
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.use_mean_joint = args.use_mean_joint

        logger.info(f't2m dataset has {self.__len__()} samples..')
        # self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        joint, valid = data['joint'], data['valid']
        joint = random.choice(joint)
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        joint = joint[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        if self.use_mean_joint:
            joint = (joint - self.mean_joint) / self.std_joint

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            joint = np.concatenate([joint,
                                     np.zeros((self.max_motion_length - m_length, joint.shape[1]))
                                     ], axis=0)
        return (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens),
                joint, valid)

class Text2MotionJointSimpleDataset(Text2MotionJointDataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mean_joint, std_joint, split='train', args=None):
        super(Text2MotionJointSimpleDataset, self).__init__(opt, mean, std, split_file, w_vectorizer, mean_joint, std_joint, split, args)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        joint, valid = data['joint'], data['valid']
        joint = random.choice(joint)
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        joint = joint[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        if self.use_mean_joint:
            joint = (joint - self.mean_joint) / self.std_joint

        if self.mask_joint:
            joint_mask = np.random.binomial(1, 1-self.joint_mask_ratio, size=(m_length, 1)).repeat(joint.shape[-1], axis=-1)
            joint_mask2 = np.random.binomial(1, 1-self.joint_mask_ratio, size=(22))
            mask_index = np.where(joint_mask2==0)
            for idx in mask_index[0]:
                joint_mask[:,self.relate_feats2d[idx]] = 0

            motion_mask = np.random.binomial(1, 1-self.joint_mask_ratio, size=(m_length, 1)).repeat(motion.shape[-1], axis=-1)
            motion_mask2 = np.random.binomial(1, 1-self.joint_mask_ratio, size=(22))
            mask_index = np.where(motion_mask2==0)
            for idx in mask_index[0]:
                motion_mask[:,self.relate_feats3d[idx]] = 0
        else:
            joint_mask = np.ones_like(joint)
            motion_mask = np.ones_like(motion)
        
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            joint = np.concatenate([joint,
                                     np.zeros((self.max_motion_length - m_length, joint.shape[1]))
                                     ], axis=0)
            joint_mask = np.concatenate([joint_mask,
                                     np.zeros((self.max_motion_length - m_length, joint_mask.shape[1]))
                                     ], axis=0)
            motion_mask = np.concatenate([motion_mask,
                                     np.zeros((self.max_motion_length - m_length, motion_mask.shape[1]))
                                     ], axis=0)
        
        return (caption, motion, m_length, joint, valid, joint_mask, motion_mask)

class HumanML3D(data.Dataset):
    def __init__(self, datapath='./data/t2m/humanml_opt.txt', split="train", args=None, **kwargs):
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        device = None  
        opt = get_opt(datapath, device)
        opt.data_root = args.data_root

        opt.meta_dir = './data/t2m'
        self.opt = opt


        self.mean = np.load(pjoin(opt.meta_dir, 'Mean.npy'))
        self.std = np.load(pjoin(opt.meta_dir, 'Std.npy'))
        
        self.mean_for_eval = np.load(pjoin(opt.meta_dir, 'mean_eval.npy'))
        self.std_for_eval = np.load(pjoin(opt.meta_dir, 'std_eval.npy'))

        self.mean_joint = np.load(pjoin(opt.meta_dir, 'Mean_complicate2d.npy'))
        self.std_joint = np.load(pjoin(opt.meta_dir, 'Std_complicate2d.npy'))


        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        self.w_vectorizer = WordVectorizer('./data/glove', 'our_vab')
        if split == 'test':
            self.t2m_dataset = Text2MotionJointDataset(self.opt, self.mean, self.std, self.split_file,
                                                    self.w_vectorizer, self.mean_joint, self.std_joint, split=split, args=args)
        elif split == 'train':
            self.t2m_dataset = Text2MotionJointSimpleDataset(self.opt, self.mean, self.std, self.split_file,
                                                    self.w_vectorizer, self.mean_joint, self.std_joint, split=split, args=args)
        elif split == 'generate':
            self.t2m_dataset = data.Dataset()
        else:
            raise NotImplementedError

        if args.local_rank != -1:
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.device = torch.device(f"cuda:0")
        self.mean_t = torch.as_tensor(self.mean).to(self.device)
        self.std_t = torch.as_tensor(self.std).to(self.device)
        self.mean_joint_t = torch.as_tensor(self.mean_joint).to(self.device)
        self.std_joint_t = torch.as_tensor(self.std_joint).to(self.device)
        self.mean_for_eval_t = torch.as_tensor(self.mean_for_eval).to(self.device)
        self.std_for_eval_t = torch.as_tensor(self.std_for_eval).to(self.device)

        if split != 'generate':
            assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                            'it is probably because your data dir has only texts and no motions.\n' \
                                            'To train and evaluate MDM you should get the FULL data as described ' \
                                            'in the README file.'

    def inv_transform(self, data):
        return data * self.std_t + self.mean_t
    
    def transform(self, data):
        return (data - self.mean_t) / self.std_t
    
    def inv_transform2d(self, data):
        return data * self.std_joint_t + self.mean_joint_t
    
    def transform2d(self, data):
        return (data - self.mean_joint_t) / self.std_joint_t

    def renorm4t2m(self, data):
        data = data * self.std_t + self.mean_t
        data = (data - self.mean_for_eval_t) / self.std_for_eval_t
        return data

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
    
