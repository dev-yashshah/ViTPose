# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import progressbar

def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
        ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
        f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    if pose_lift_dataset == 'Body3DH36MDataset':
        if pose_det_dataset in ['TopDownH36MDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
            # rearrange other keypoints
            keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
        if pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

            if 'PoseTrack18' in pose_det_dataset:
                keypoints_new[0] = keypoints[1]
                # don't extrapolate the head top confidence score
                keypoints_new[16, 2] = keypoints_new[0, 2]
            else:
                # head top is extrapolated from neck and head
                keypoints_new[0] = (4 * keypoints_new[16] -
                                    keypoints_new[1]) / 3
                # don't extrapolate the head top confidence score
                keypoints_new[0, 2] = keypoints_new[16, 2]
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
            ]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # head top is head top
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is neck
            keypoints_new[1] = keypoints[13]
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[0:12]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # head top is top_head
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
            ]]

        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


def main():
    parser = ArgumentParser()
    parser.add_argument('--det_config', 
        default='mmdet/configs/faster_rcnn_r50_fpn_coco.py',
        help='Config file for detection')
    parser.add_argument('--det_checkpoint', 
        default='mmdet/pretrained_weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', 
        help='Checkpoint file for detection')
    parser.add_argument(
        '--pose_detector_config',
        type=str,
        default='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/ViTPose_huge_crowdpose_256x192.py',
        #default='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py',
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        '--pose_detector_checkpoint',
        type=str,
        default='pretrained_weights/vitpose-h-multi-crowdpose.pth',
        #default='pretrained_weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        '--pose_lifter_config', 
        #default='configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py',
        default='configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--pose_lifter_checkpoint',
        default='configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--img-path', type=str, default='', help='Image folder')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='vis_results',
        help='Root of the output image file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.9,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the 2D pose estimation '
        'results. See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')

    print('Initializing model...')
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_det_model, TopDown), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running 2D pose detection inference...')
    files=os.listdir(args.img_path)
    for i in progressbar.progressbar(range(len(files))):
        pose_det_results_last = pose_det_results
        img=cv2.imread(osp.join(args.img_path,files[i]))
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, img)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)
        
        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_results_last,
            img,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    print('Initializing model...')
    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lift_model, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    
    os.makedirs(args.out_img_root, exist_ok=True)

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(
                keypoints, pose_det_dataset, pose_lift_dataset)

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    # build pose smoother for temporal refinement
    if args.smooth:
        smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2)
    else:
        smoother = None

    num_instances = args.num_instances
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)
    
    print('Running 2D-to-3D pose lifting inference...')
    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        # extract and pad input pose2d sequence
        img=cv2.imread(osp.join(args.img_path,files[i]))
        pose_results_2d = extract_pose_sequence(
            pose_det_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=data_cfg.seq_frame_interval)

        # smooth 2d results
        if smoother:
            pose_results_2d = smoother.smooth(pose_results_2d)

        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            with_track_id=False,
            image_size=[img.shape[1],img.shape[0]],
            norm_pose_2d=args.norm_pose_2d)

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results[idx]
            instance_id = 0 #det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            res['bbox'] = det_res['bbox']
            res['track_id'] = instance_id
            pose_lift_results_vis.append(res)

        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=cv2.imread(osp.join(args.img_path,files[i])),
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            out_file=None,
            radius=args.radius,
            thickness=args.thickness,
            num_instances=num_instances,
            show=args.show)

        cv2.imwrite(osp.join(args.out_img_root,f'vis_{files[i]}'),img_vis)

if __name__ == '__main__':
    main()