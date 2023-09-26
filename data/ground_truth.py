import numpy as np
import sys

sys.path.append('../')

from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from utils.utils import wrap

file_name = "data_3d_h36m"
op_file_name_2d = "data_2d_h36m_gt_np"
op_file_name_3d = "data_3d_h36m_gt_np"

dataset = Human36mDataset(file_name + '.npz')
output_2d_poses = {}
output_3d_poses = {}
for subject in dataset.subjects():
    output_2d_poses[subject] = {}
    output_3d_poses[subject] = {}
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        positions_2d = []
        positions_3d = []
        for cam in anim['cameras']:
            pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
            positions_3d.append(pos_3d)
            pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
            pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
            positions_2d.append(pos_2d_pixel_space.astype('float32'))
        output_2d_poses[subject][action] = positions_2d
        output_3d_poses[subject][action] = positions_3d
metadata = {
    'num_joints': dataset.skeleton().num_joints(),
    'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
}

print("Saving Ground Truth")
np.savez_compressed(op_file_name_2d, positions_2d = output_2d_poses, metadata = metadata)
np.savez_compressed(op_file_name_3d, positions_3d = output_3d_poses, metadata = metadata)