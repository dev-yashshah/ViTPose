# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmpose.datasets import DatasetInfo

from mmdet.apis import (inference_detector, init_detector, show_result_pyplot)
import progressbar

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-path', type=str, help='Image path')
    parser.add_argument(
        '--out-img-root',
        default='',
        help='Root of the output image file. '
        'Default not saving the visualization image.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    os.makedirs(args.out_img_root, exist_ok=True)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    for img_path in progressbar.progressbar(os.listdir(args.img_path)):
        # keep the person class bounding boxes.
        img=cv2.imread(os.path.join(args.img_path,img_path))
        size=(img.shape[1],img.shape[0])
        
        mmdet_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
        # test a single image
        mmdet_result = inference_detector(mmdet_model, img)
        #print(mmdet_result)
        #break
        
        #person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]
        person_results=process_mmdet_results(mmdet_result, 1) # 1 is default for person. Returns bounding boxes for all humans
        
        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            bbox_thr=0.4, #args.bbox_thr, #setting to 0.4 as a constant for now
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        with open(os.path.join(args.out_img_root,img_path.split(".")[0]+"_results.txt"),"w") as writer:
            writer.writelines(str(pose_results))
        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
            
        cv2.imwrite(os.path.join(args.out_img_root, "vis_"+img_path),vis_img)


if __name__ == '__main__':
    main()
