import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



def t2m_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)

    databatch = [torch.tensor(b[4].T).float().unsqueeze(1) for b in notnone_batches]
    jointbatch = [torch.tensor(b[7].T).float().unsqueeze(1) for b in notnone_batches]
    lenbatch = [b[5] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    jointbatchTensor = collate_tensors(jointbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)

    word_embeddings = torch.stack([torch.as_tensor(b[0]).float() for b in notnone_batches], dim=0)
    pos_one_hots = torch.stack([torch.as_tensor(b[1]).float() for b in notnone_batches], dim=0)

    new_batch = {'motion':databatchTensor,
             'mask':maskbatchTensor,
             'lengths': lenbatchTensor,
             'joint': jointbatchTensor,
             'text': [b[2] for b in notnone_batches],
             'tokens': [b[6] for b in notnone_batches],
             'valid': torch.as_tensor([b[8] for b in notnone_batches]),
             'word_embeddings': word_embeddings,
             'pos_one_hots': pos_one_hots,
             'sent_len': torch.as_tensor([b[3] for b in notnone_batches]),
            }

    return new_batch

def kit_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)

    databatch = [torch.tensor(b[4].T).float().unsqueeze(1) for b in notnone_batches]
    lenbatch = [b[5] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)

    word_embeddings = torch.stack([torch.as_tensor(b[0]).float() for b in notnone_batches], dim=0)
    pos_one_hots = torch.stack([torch.as_tensor(b[1]).float() for b in notnone_batches], dim=0)

    new_batch = {'motion':databatchTensor,
             'mask':maskbatchTensor,
             'lengths': lenbatchTensor,
             'text': [b[2] for b in notnone_batches],
             'tokens': [b[6] for b in notnone_batches],
             'word_embeddings': word_embeddings,
             'pos_one_hots': pos_one_hots,
             'sent_len': torch.as_tensor([b[3] for b in notnone_batches]),
            }

    return new_batch

def simple_collate(batch):
    notnone_batches = [b for b in batch if b is not None]

    databatch = [torch.tensor(b[1].T).float().unsqueeze(1) for b in notnone_batches]
    jointbatch = [torch.tensor(b[3].T).float().unsqueeze(1) for b in notnone_batches]
    lenbatch = [b[2] for b in notnone_batches]
    jointmaskbatch = [torch.tensor(b[5].T).float().unsqueeze(1) for b in notnone_batches]
    motionmaskbatch = [torch.tensor(b[6].T).float().unsqueeze(1) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    jointbatchTensor = collate_tensors(jointbatch)
    jointmaskbatchTensor = collate_tensors(jointmaskbatch)
    motionmaskbatchTensor = collate_tensors(motionmaskbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)


    new_batch = {'motion':databatchTensor,
             'mask':maskbatchTensor,
             'lengths': lenbatchTensor,
             'joint': jointbatchTensor,
             'joint_mask': jointmaskbatchTensor,
             'motion_mask': motionmaskbatchTensor,
             'text': [b[0] for b in notnone_batches],
             'valid': torch.as_tensor([b[4] for b in notnone_batches]),
            }

    return new_batch
