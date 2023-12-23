import torch
import clip
import numpy as np
from scipy import linalg
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed import ReduceOp

from diffusion import logger
from utils.load_utils import encode_text
from model.cfg_sampler import ClassifierFreeSampleModel

def p_variance(x_0, x_t, t, diffusion):
    noise = torch.randn_like(x_0)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_0.shape) - 1)))

    model_mean, _, _ = diffusion.q_posterior_mean_variance(
        x_start=x_0, x_t=x_t, t=t
    )
    model_log_variance = torch.from_numpy(diffusion.posterior_log_variance_clipped).to(device=x_0.device)[t].float()
    model_log_variance = model_log_variance.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(x_t.shape)

    x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    return x_t

def gather_dist(g_example):
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(g_example) for _ in range(world_size)]
    dist.all_gather(tensor_list, g_example)
    g_example = torch.cat(tensor_list)
    return g_example

def mean_dist(m_example):
    world_size = dist.get_world_size()
    m_example = torch.tensor(m_example, dtype=torch.float32).cuda()
    dist.all_reduce(m_example, op=ReduceOp.SUM)
    m_example = m_example / world_size
    return m_example.cpu().numpy()

@torch.no_grad()   
def evaluation_transformer(args, val_loader, model, diffusion,
                            clip_model, eval_wrapper, device, nb_iter,
                            test_limit): 

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    if args.local_rank < 1:
        bar = enumerate(tqdm(val_loader))
    else:
        bar = enumerate(val_loader)
    for i, batch in bar:
        if i >= test_limit:
            break
        motion = batch['motion'].to(device)
        bs = motion.shape[0]
        batch['enc_text'] = encode_text(clip_model, batch['text'], device)
        if args.classifier_free:
            batch['scale'] = torch.ones(motion.shape[0],
                                                    device=device) * args.guidance_param
            
        word_embeddings = batch['word_embeddings']
        pos_one_hots = batch['pos_one_hots']
        sent_len = batch['sent_len']
        m_length = batch['lengths']
        
        if args.test_generatefrom2d:
            indices = list(range(1000))[::-1]
            if args.local_rank < 1:
                indices = tqdm(indices)
            motion_t = torch.randn((bs, 263, 1, 196), dtype=torch.float32, device=device)
            joint_t = torch.randn((bs, 134, 1, 196), dtype=torch.float32, device=device)

            for i in indices:
                t = torch.tensor([i] * bs, device=device)

                if i > args.change_idx:
                    joint_predict = model.model(joint_t, t, batch['enc_text'], return_m=False, return_j=True)
                    joint_t = p_variance(joint_predict['j'], joint_t, t, diffusion)

                elif i == args.change_idx:
                    joint_predict = model.model(joint_t, t, batch['enc_text'], return_m=True, return_j=False)
                    if i == 0:
                        motion_t = joint_predict['m']
                        break
                    noise_motion = torch.randn_like(joint_predict['m'])
                    motion_t = diffusion.q_sample(joint_predict['m'], t - 1, noise=noise_motion)
                    # motion_t = joint_predict['m']
                
                else:
                    motion_predict = model.model(motion_t, t, batch['enc_text'], return_m=True, return_j=False)
                    motion_t = p_variance(motion_predict['m'], motion_t, t, diffusion)


            pred_motion = motion_t
            
        else:
            pred_motion = diffusion.p_sample_loop(
                            model,
                            motion.shape,
                            clip_denoised=False,
                            model_kwargs=batch,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,)
        
        motion = motion[:,:,0].permute(0,2,1)
        pred_motion = pred_motion[:,:,0].permute(0,2,1)
        if args.eval_upper_lower == 0:
            motion = val_loader.dataset.renorm4t2m(motion)
            pred_motion = val_loader.dataset.renorm4t2m(pred_motion)
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
        elif args.eval_upper_lower == 1:
            et, em = eval_wrapper.get_co_embeddings_upper(word_embeddings, pos_one_hots, sent_len, motion, m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings_upper(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
        else:
            et, em = eval_wrapper.get_co_embeddings_lower(word_embeddings, pos_one_hots, sent_len, motion, m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings_lower(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0)
    motion_pred_np = torch.cat(motion_pred_list, dim=0)

    if args.local_rank != -1:
        motion_annotation_np = gather_dist(motion_annotation_np)
        motion_pred_np = gather_dist(motion_pred_np)

    motion_annotation_np = motion_annotation_np.cpu().numpy()
    motion_pred_np = motion_pred_np.cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 10)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if args.local_rank != -1:
        R_precision_real = mean_dist(R_precision_real)
        R_precision = mean_dist(R_precision)
        matching_score_real = mean_dist(matching_score_real)
        matching_score_pred = mean_dist(matching_score_pred)



    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f},\n \
R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    return {'fid': fid, 'diversity_real': diversity_real, 'diversity': diversity,
                'R_precision_real': R_precision_real, 'R_precision': R_precision,
                'matching_score_real': matching_score_real, 'matching_score_pred': matching_score_pred}

    

@torch.no_grad()        
def evaluation_transformer_mm(args, val_loader, model, diffusion,
                            clip_model, eval_wrapper, device, nb_iter,
                            test_limit): 

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    

    motion_pred_list = []


    if args.local_rank < 1:
        bar = enumerate(tqdm(val_loader))
    else:
        bar = enumerate(val_loader)
    for i, batch in bar:
        if i >= test_limit:
            break
        
        motion = batch['motion']
        bs = motion.shape[0]
        batch['enc_text'] = encode_text(clip_model, batch['text'], device)
        if args.classifier_free:
            batch['scale'] = torch.ones(motion.shape[0],
                                                    device=device) * args.guidance_param
            
        word_embeddings = batch['word_embeddings']
        pos_one_hots = batch['pos_one_hots']
        sent_len = batch['sent_len']
        m_length = batch['lengths']
        
        motion_pred_list_batch = []
        for j in range(30):
            pred_motion = diffusion.p_sample_loop(
                            model,
                            motion.shape,
                            clip_denoised=False,
                            model_kwargs=batch,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,)
            
            pred_motion = pred_motion[:,:,0].permute(0,2,1)
            pred_motion = val_loader.dataset.renorm4t2m(pred_motion)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
            motion_pred_list_batch.append(em_pred.unsqueeze(1))
        motion_pred_list_batch = torch.cat(motion_pred_list_batch, dim=1)
        motion_pred_list.append(motion_pred_list_batch)

    motion_pred_np = torch.cat(motion_pred_list, dim=0)

    if args.local_rank != -1:
        motion_pred_np = gather_dist(motion_pred_np)

    motion_pred_np = motion_pred_np.cpu().numpy()
    multimodality = calculate_multimodality(motion_pred_np, 10)


    msg = f"--> \t Eva. Iter {nb_iter} :, multimodality {multimodality}"
    logger.info(msg)
    
    return {'multimodality': multimodality}

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), dist_mat.trace()
    else:
        return top_k_mat, np.diagonal(dist_mat)

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist

