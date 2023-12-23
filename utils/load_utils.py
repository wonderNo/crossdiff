import os
from diffusion import logger
import clip
import torch

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def masked_l2(a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
    loss = (a - b) ** 2
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    if mask.shape[1] == 1:
        n_entries = a.shape[1] * a.shape[2]
    else:
        n_entries = 1
    non_zero_elements = sum_flat(mask) * n_entries
    mse_loss_val = loss / (non_zero_elements + 1e-8)
    return mse_loss_val


def l2(a, b):
    loss = (a - b) ** 2
    loss = sum_flat(loss)  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1] * a.shape[2] * a.shape[3]
    mse_loss_val = loss / (n_entries + 1e-8)
    return mse_loss_val


def find_resume_checkpoint(dir):
    if not os.path.exists(dir):
        return '', -1
    checkpoints = sorted(os.listdir(dir),
                                 key=lambda x: int(x[5:-3]),
                                 reverse=True)
    if len(checkpoints) == 0:
        return '', -1
    else:
        start_epoch = int(checkpoints[0][5:9])
        return os.path.join(dir, checkpoints[0]), start_epoch


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def load_and_freeze_clip():
    clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu',
                                            jit=False, download_root='/apdcephfs_cq3/share_1290939/zepingren/CLIP')  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(clip_model, raw_text, device):
    # raw_text - list (batch_size length) of strings with input text prompts
    max_text_len = 20  # Specific hardcoding for humanml dataset
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2 # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        # print('texts', texts.shape)
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
        # print('texts after pad', texts.shape, texts)
    else:
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float()