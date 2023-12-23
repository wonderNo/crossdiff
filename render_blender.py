import shutil
import os
import sys
sys.path.insert(0,'/root/.local/lib/python3.9/site-packages')


try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

from visualize.blender.render import render
from visualize.blender.video import Video
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", '-f', required=True, type=str)
    args = parser.parse_args()

    print(f'visualizing file {args.file}...')
    data = np.load(args.file)
    frames_folder = args.file[:-4] + '_frames'
    mp4_file = args.file[:-4] + '_blender.mp4'
    mode = 'video' #'sequence' #'video' 'frame
    if mode == 'video':
        frames_folder = args.file[:-4] + '_frames'
        mp4_file = args.file[:-4] + '_blender.mp4'
    elif mode == 'sequence':
        frames_folder = args.file[:-4] + '_blender.png'
    elif mode == 'frame':
        frames_folder = args.file[:-4] + '_blender2.png'


    render(data, frames_folder, mode=mode,)

    if mode == 'video':
        video = Video(frames_folder, fps=20)
        video.save(out_path=mp4_file)
        shutil.rmtree(frames_folder)
        print(f"remove tmp fig folder and save video in {mp4_file}")

if __name__ == '__main__':
    main()
