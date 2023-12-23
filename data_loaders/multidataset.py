import numpy as np
from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.ucf101 import UCF101

class MultiDataset(HumanML3D):

    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", args=None):
        super(MultiDataset, self).__init__(mode, datapath, split, args)

        self.ucf101dataset = UCF101(args)

        self.cut_length = len(self.ucf101dataset)
        self.little = args.ucf_ratio > 0

        if self.little:
            self.length = int(len(self.ucf101dataset) / args.ucf_ratio)
        else:
            self.length = len(self.t2m_dataset) + len(self.ucf101dataset)


    def __getitem__(self, index):
        if index < self.cut_length:
            return self.ucf101dataset[index]
        elif self.little:
            return self.t2m_dataset[np.random.randint(0, len(self.t2m_dataset))]
        else:
            return self.t2m_dataset[index - self.cut_length]

    def __len__(self):
        return self.length