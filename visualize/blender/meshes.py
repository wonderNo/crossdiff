import numpy as np
import matplotlib

from .materials import body_material

# green
GT_SMPL = body_material(0.0, 0.392, 0.158)

# blue
GEN_SMPL = body_material(0.022, 0.129, 0.439)


class Meshes:
    def __init__(self, data, *, ours, mode, canonicalize, always_on_floor, oldrender=True, **kwargs):
        data = prepare_meshes(data, canonicalize=canonicalize,
                              always_on_floor=always_on_floor)

        self.faces = np.load('./visualize/joints2smpl/smpl_models/faces.npy')
        self.data = data
        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)

        if ours:
            self.mat = GT_SMPL
        else:
            self.mat = GEN_SMPL

    def get_sequence_mat(self, frac, ours):
        if ours:
            cmap = matplotlib.cm.get_cmap('Greens')
        else:
            cmap = matplotlib.cm.get_cmap('Blues')
            
        # cmap = matplotlib.cm.get_cmap('Oranges')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        # mat = body_material(156/255, 156/255, 156/255)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fitted mesh do not need fixing axis
    # # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data
