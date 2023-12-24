from torch.utils.data import DataLoader
from data_loaders.tensors import t2m_collate, simple_collate
from diffusion import logger

from torch.utils.data.distributed import DistributedSampler

def get_dataset_class(name):
    if name in ["humanml"]:
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == 'multi':
        from data_loaders.multidataset import MultiDataset
        return MultiDataset
    elif name == 'ufc':
        from data_loaders.ucf101 import UFC101
        return UFC101
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(split='train'):
    if split == 'train':
        return simple_collate
    else:
        return t2m_collate


def get_dataset(name, split='train', args=None):
    DATA = get_dataset_class(name)
    if name in ["humanml", "multi"]:
        dataset = DATA(split=split, args=args)
    elif name in ['ufc']:
        dataset = DATA(args)
   
    return dataset


def get_dataset_loader(name, batch_size=1, split='train', args=None):
    dataset = get_dataset(name, split, args=args)
    collate = get_collate_fn(split)

    if split == 'generate':
        return dataset

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=8, drop_last=True, collate_fn=collate,
        sampler=sampler
        )
        logger.info(f'{name} has {len(dataset)} samples with {len(loader)} batch in ddp mode..')
    else:
        loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate,
        )

        logger.info(f'{name} has {len(dataset)} samples with {len(loader)} batch in single gpu mode..')

    

    
    return loader