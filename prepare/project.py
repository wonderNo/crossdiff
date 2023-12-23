from tqdm import tqdm
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3




def easy_project(joints3d, cam_t, tan_fov):
    joints3d_cam = joints3d + cam_t
    joints2d = - joints3d_cam[...,:2] / (joints3d_cam[...,2:] * tan_fov)
    return joints2d

def project_np(joints3d, cam_t, tan_fov):
    return easy_project(torch.from_numpy(joints3d), cam_t, tan_fov).numpy()

def plot_2d_motion(save_path, joints, figsize=(10, 10), fps=20, radius=4, kinematic_tree=None):
    fig, ax = plt.subplots()
    # MINS = joints.min(axis=(0,1))
    # MAXS = joints.max(axis=(0,1))
    MINS = [-1, -1]
    MAXS = [1, 1]

    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = joints.shape[0]
    def update(frame):
        ax.clear()
        ax.set_xlim(MINS[0], MAXS[0])
        ax.set_ylim(MINS[1], MAXS[1])
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot(joints[frame, chain, 0], joints[frame, chain, 1], linewidth=linewidth, color=color)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion(save_path, kinematic_tree, joints, figsize=(10, 10), fps=20, radius=4):
#     matplotlib.use('Agg')

    # title_sp = title.split(' ')
    # if len(title_sp) > 10:
        # title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        # fig.suptitle(title, fontsize=20)
        # ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'yellow','green','red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    # height_offset = MINS[1]
    # data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.clear()
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

def get_matrix_from_vec(vec):
    # (B,2)
    vec = vec / np.sqrt(np.sum(vec ** 2, axis=-1, keepdims=True))
    mat = np.stack([vec[:, 0], -vec[:,1], vec[:,1], vec[:,0]], axis=1).reshape((-1,2,2))
    return mat

def process_file(positions, feet_thre=0.002):
    # (seq_len, joints_num, 2)
    fid_r, fid_l = [8, 11], [7, 10]
    # fid_r, fid_l = [14, 15], [19, 20] #KIT
    T = positions.shape[0]
    positions = positions - positions[:1,:1]

    global_positions = positions.copy()


    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l = ((feet_l_x + feet_l_y) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r = (((feet_r_x + feet_r_y) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)



    rot_params = np.zeros((positions.shape[0], positions.shape[1] - 1, 2, 2))
    for chain in kinematic_chain:
        R = np.eye(2)[None,].repeat(len(positions), axis=0)
        for j in range(len(chain) - 1):
            # (batch, 3)
            v = positions[:, chain[j+1]] - positions[:, chain[j]]
            rot_mat = get_matrix_from_vec(v)

            R_loc = np.einsum('bij,bjk->bik', R.transpose(0,2,1), rot_mat)

            rot_params[:,chain[j + 1] - 1] = R_loc
            R = rot_mat

    cont_2d_params = rot_params[:,:,0]

    root_v = global_positions[1:,0] - global_positions[:1,0]

    positions -= positions[:, 0:1]
    positions = positions[:,1:].reshape(T, -1)

    rot_data = cont_2d_params.reshape(T, -1)

    local_vel = global_positions[1:] - global_positions[:-1]
    local_vel = local_vel.reshape(len(local_vel), -1)

    # root_v:2, loc_position:21*2, rot_2d:21*2, local_vel:22*2, feet_contact:4--[B,134]
    data = np.concatenate([root_v, positions[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)

    return data

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_root", required=True, type=str)

    args = parser.parse_args()
    joint3d_dir = os.path.join(args.data_root, 'new_joints')
    joint2d_dir = os.path.join(args.data_root, 'new_joints2d')
    movie_dir = os.path.join(args.data_root, 'animations')
    joint2d_complicate_dir = os.path.join(args.data_root, 'new_joints2d_complicate')

    # kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] kit
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    FOV = 10
    tan_fov = np.tan(np.radians(FOV/2.))
    cam_t = torch.tensor([0, -1, -1 / tan_fov-10], dtype=torch.float32)
    # cam_t = torch.tensor([0, -1, -1 / tan_fov-10], dtype=torch.float32) * 520
    Rs = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]], [[0,0,1],[0,1,0],[-1,0,0]],
                 [[-1,0,0],[0,1,0],[0,0,-1]],[[0,0,-1],[0,1,0],[1,0,0]]],dtype=torch.float32)

    os.makedirs(joint2d_dir, exist_ok=True)
    os.makedirs(movie_dir, exist_ok=True)

    files = os.listdir(joint3d_dir)
    try:
        for i, file in enumerate(tqdm(files)):
            joint3d_file = os.path.join(joint3d_dir, file)
            

            joints3d = np.load(joint3d_file)
            joints3d = torch.from_numpy(joints3d)
            for j,R in enumerate(Rs):
                # project
                save_file = os.path.join(joint2d_dir, file[:-4] + f'-{j}.npy')
                save_file_complicate = os.path.join(joint2d_complicate_dir, file[:-4] + f'-{j}.npy')
                if os.path.exists(save_file_complicate):
                    continue
                if len(joints3d.shape) != 3:
                    print(f'wrong in {joint3d_file}')
                    np.save(save_file, np.zeros((1,21,2), dtype=np.float32))
                    continue
                joint3d = torch.einsum('tjk,pk->tjp', joints3d, R)
                joints2d = easy_project(joint3d, cam_t, tan_fov).numpy()
                joints2d = joints2d - joints2d[:1,:1]
                if i < 10:
                    if j == 0:
                        movie3d_file = os.path.join(movie_dir, file[:-4] + '-3d.mp4')
                        plot_3d_motion(movie3d_file, kinematic_chain, joints3d.numpy())
                    movie2d_file = os.path.join(movie_dir, file[:-4] + f'-2d-{j}.mp4')
                    plot_2d_motion(movie2d_file, joints2d, kinematic_tree=kinematic_chain)
                np.save(save_file, joints2d)


                # process 
                if joints2d.shape[1] != 22:
                    raise NameError('not 22 joint')
                if len(joints2d) == 1:
                    # data = np.zeros((1,128), dtype=np.float32)
                    joints2d_complicate = np.zeros((1,134), dtype=np.float32)
                else:
                    joints2d_complicate = process_file(joints2d)
                np.save(save_file_complicate, joints2d_complicate.astype(np.float32))
                
    except Exception as e:
        print(file)
        print(e)


    print('all done!')
