import numpy as np
from visualize.simplify_loc2rot import joints2smpl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", '-f',required=True, type=str)
args = parser.parse_args()
print(f'dealing file {args.file}')
motion = np.load(args.file)

j2s = joints2smpl(num_frames=motion.shape[0], device_id=0)

meshes, vertices = j2s.joint2smpl(motion)
np.save(args.file[:-4] + '_mesh.npy', vertices)

print('done')