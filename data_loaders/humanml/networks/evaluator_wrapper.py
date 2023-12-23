from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin


def build_evaluators(opt):
    

    if opt['eval_part'] == 'all':
        eval_index = list(range(opt['dim_pose']-4))
    else:
        lower_joint = np.array([1,2,4,5,7,8,10,11]) - 1
        lower_joint2 = np.array([0, 1,2,4,5,7,8,10,11])
        lower_index1 = np.array([0,1,2,3])
        lower_index2 = np.stack([4 + lower_joint * 3, 5 + lower_joint * 3, 6 + lower_joint * 3], axis=1).reshape(-1)
        lower_index3 = np.stack([67 + lower_joint * 6, 68 + lower_joint * 6, 69 + lower_joint * 6,
                                    70 + lower_joint * 6, 71 + lower_joint * 6, 72 + lower_joint * 6,], axis=1).reshape(-1)
        lower_index4 = np.stack([193 + lower_joint2 * 3, 194 + lower_joint2 * 3, 195 + lower_joint2 * 3], axis=1).reshape(-1)
        lower_index5 = np.array([259,260,261,262])
        lower_index = np.concatenate([lower_index1, lower_index2, lower_index3, lower_index4, lower_index5]).tolist() # 107
        upper_index = [i for i in range(263) if i not in lower_index] # 156
        lower_index = lower_index[:-4] # 103

        if opt['eval_part'] == 'upper':
            eval_index = upper_index
        elif opt['eval_part'] == 'lower':
            eval_index = lower_index
        else:
            raise NotImplementedError('Unsupported evaluation part.')

    movement_enc = MovementConvEncoder(len(eval_index), opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])


    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], 't2m', opt["eval_part"] + '.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc, eval_index

class EvaluatorWrapper(object):

    def __init__(self, device, eval_part='all'):
        opt = {
            'dataset_name': 'humanml',
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': './data',
            'unit_length': 4,
            'eval_part': eval_part
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder, self.eval_index = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions[..., self.eval_index].detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = torch.div(m_lens, self.opt['unit_length'], rounding_mode='floor')
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding
    

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions[..., self.eval_index].detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
    
