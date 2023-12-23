import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

def plot_2d_motion(save_path, kinematic_tree, joints, title='', figsize=(10, 10), fps=20, kp_thr=0.3):
    fig, ax = plt.subplots()
    title = '\n'.join(wrap(title, 20))


    # MINS = joints.min(axis=(0,1))
    # MAXS = joints.max(axis=(0,1))
    # width = (MAXS - MINS).max() / 2
    # middle = (MAXS + MINS) / 2
    # MAXS = middle + width
    # MINS = middle - width
    MINS = [-1, -1]
    MAXS = [1, 1]

    def init():
        ax.set_xlim(MINS[0], MAXS[0])
        ax.set_ylim(MINS[1], MAXS[1])
        fig.suptitle(title, fontsize=10)
    
    colors = ["#DD5A37", # 红
            "#D69E00", 
            "#ef1f9b",
            "#DD5A37", 
            "#D69E00", 
            '#a73dfa', # 紫
            '#fca6f7', # 粉
            ]
    # colors = ['red', 'blue', 'black', 'red', 'blue',  
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #          'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = joints.shape[0]
    def update(frame):
        ax.clear()
        init()
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 2.0
            if joints.shape[2] == 3:
                for j in range(len(chain) - 1):
                    if joints[frame, chain[j], 2] > kp_thr and joints[frame, chain[j+1], 2] > kp_thr:
                        ax.plot(joints[frame, chain[j:j+2], 0], joints[frame, chain[j:j+2], 1], linewidth=linewidth, color=color)
            else:
                ax.plot(joints[frame, chain, 0], joints[frame, chain, 1], linewidth=linewidth, color=color)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title='', dataset=None, figsize=(10, 10), fps=20, radius=4,
                   vis_mode='default', gt_frames=[], view_angle=(120, -90)):
    matplotlib.use('Agg')

    if os.path.isdir(save_path):
        title = ''
    else:
        title = '\n'.join(wrap(title, 20))

    def init():
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

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

    # preparation related to specific datasets
    data *= 1.3  # scale for visualization


    fig = plt.figure(figsize=figsize)
    # plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", # 红
                    "#D69E00", # 黄
                    "#b7fa3d", # 绿
                    "#3dfae3", # 浅蓝
                    "#3d55fa", # 深蓝
                    '#a73dfa', # 紫
                    '#fca6f7', # 粉
                    ]  # Generation color
    colors = ["#DD5A37", # 红
            "#D69E00", 
            "#ef1f9b",
            "#DD5A37", 
            "#D69E00", 
            '#a73dfa', # 紫
            '#fca6f7', # 粉
            ]
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    # height_offset = MINS[1]
    # data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 1] -= data[:, 0:1, 1]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.clear()
        init()
        # ax.view_init(elev=120, azim=-90)
        ax.view_init(*view_angle)
        ax.dist = 14
        #         ax =
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #              MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)
    if os.path.isdir(save_path):
        for i in range(len(list(ani.new_frame_seq()))):
            # ani.save(os.path.join(save_path,f"frame_{i}.png"), writer='pillow', savefig_kwargs={'facecolor': 'white'})
            ani._draw_next_frame(i, blit=False)
            ax.figure.savefig(os.path.join(save_path,f"frame_{i}.png"))
    else:
        ani.save(save_path, fps=fps)

    plt.close()

if __name__ == '__main__':
    npy_file = '/apdcephfs_cq3/share_1290939/zepingren/ufc101/annot2/t2m2/v_BaseballPitch_g09_c02.npy'
    motion = np.load(npy_file)
    skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    # save_dir = npy_file[:-4]
    # os.makedirs(save_dir,exist_ok=True)
    # plot_3d_motion(save_dir, skeleton,motion)
    # plot_3d_motion('test.mp4', skeleton,motion)
    plot_2d_motion('test.mp4', skeleton, motion, kp_thr=-1)
    # plot_3d_motion('test.mp4', skeleton,motion,view_angle=(180,-30))