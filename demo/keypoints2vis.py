import json
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def imshow_keypoints_3d(
    pose_result,
    img=None,
    skeleton=None,
    pose_kpt_color=None,
    pose_link_color=None,
    vis_height=400,
    kpt_score_thr=0.3,
    num_instances=-1,
    *,
    axis_azimuth=70,
    axis_limit=1.7,
    axis_dist=10.0,
    axis_elev=15.0,
):
    """Draw 3D keypoints and links in 3D coordinates.

    Args:
        pose_result (list[dict]): 3D pose results containing:
            - "keypoints_3d" ([K,4]): 3D keypoints
            - "title" (str): Optional. A string to specify the title of the
                visualization of this pose result
        img (str|np.ndarray): Opptional. The image or image path to show input
            image and/or 2D pose. Note that the image should be given in BGR
            channel order.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.ndarray[Nx3]`): Color of N keypoints. If None, do
            not nddraw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links. If None, do not
            draw links.
        vis_height (int): The image height of the visualization. The width
                will be N*vis_height depending on the number of visualized
                items.
        kpt_score_thr (float): Minimum score of keypoints to be shown.
            Default: 0.3.
        num_instances (int): Number of instances to be shown in 3D. If smaller
            than 0, all the instances in the pose_result will be shown.
            Otherwise, pad or truncate the pose_result to a length of
            num_instances.
        axis_azimuth (float): axis azimuth angle for 3D visualizations.
        axis_dist (float): axis distance for 3D visualizations.
        axis_elev (float): axis elevation view angle for 3D visualizations.
        axis_limit (float): The axis limit to visualize 3d pose. The xyz
            range will be set as:
            - x: [x_c - axis_limit/2, x_c + axis_limit/2]
            - y: [y_c - axis_limit/2, y_c + axis_limit/2]
            - z: [0, axis_limit]
            Where x_c, y_c is the mean value of x and y coordinates
        figsize: (float): figure size in inch.
    """

    show_img = img is not None
    num_instances = len(pose_result)
    num_axis = num_instances + 1 if show_img else num_instances
    img_h, img_w, _ = img.shape

    plt.ioff()
    fig = plt.figure(figsize=(vis_height * num_axis * 0.01, vis_height * 0.01))

    if show_img:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale=vis_height / h
        new_size = int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

        img = cv2.resize(img, new_size)

        ax_img = fig.add_subplot(1, num_axis, 1)
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)
        ax_img.set_axis_off()
        ax_img.set_title('Input')
        ax_img.imshow(img, aspect='equal')

    for idx, res in enumerate(pose_result):
        dummy = len(res) == 0
        kpts = np.zeros((1, 3)) if dummy else res['keypoints_3d']
        kpts_2d = np.zeros((1, 3)) if dummy else res['keypoints']
        
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        valid = kpts[:, 3] >= kpt_score_thr
        
        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts_2d[sk[0], 0]), int(kpts_2d[sk[0], 1]))
            pos2 = (int(kpts_2d[sk[1], 0]), int(kpts_2d[sk[1], 1]))

            if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                    or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                    or pos2[1] <= 0 or pos2[1] >= img_h
                    or kpts_2d[sk[0], 2] < kpt_score_thr
                    or kpts_2d[sk[1], 2] < kpt_score_thr
                    or pose_link_color[sk_id] is None):
                # skip the link that should not be drawn
                valid[sk_id + 1]=False
                continue
        
        ax_idx = idx + 2 if show_img else idx + 1
        ax = fig.add_subplot(1, num_axis, ax_idx, projection='3d')
        ax.view_init(
            elev=axis_elev,
            azim=axis_azimuth,
        )
        x_c = np.mean(kpts[valid, 0]) if sum(valid) > 0 else 0
        y_c = np.mean(kpts[valid, 1]) if sum(valid) > 0 else 0
        ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
        ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
        ax.set_zlim3d([0, axis_limit])
        ax.set_aspect('auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = axis_dist

        if not dummy and pose_kpt_color is not None:
            pose_kpt_color = np.array(pose_kpt_color)
            assert len(pose_kpt_color) == len(kpts)
            x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)
            # matplotlib uses RGB color in [0, 1] value range
            _color = pose_kpt_color[..., ::-1] / 255.
            ax.scatter(
                x_3d[valid],
                y_3d[valid],
                z_3d[valid],
                marker='o',
                color=_color[valid],
            )
        
        for pnt_cnt, points in enumerate(zip(x_3d[valid],y_3d[valid], z_3d[valid])):
            ax.text(points[0][0] * (1 + 0.01), 
                    points[1][0] * (1 + 0.01), 
                    points[2][0] * (1 + 0.01), 
                    s=str(pnt_cnt), 
                    fontsize=12)
            
        if not dummy and skeleton is not None and pose_link_color is not None:
            pose_link_color = np.array(pose_link_color)
            assert len(pose_link_color) == len(skeleton)
            
            for link, link_color in zip(skeleton, pose_link_color):
                link_indices = [_i for _i in link]
                pos1 = (int(kpts_2d[link[0], 0]), int(kpts_2d[link[0], 1]))
                pos2 = (int(kpts_2d[link[1], 0]), int(kpts_2d[link[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts_2d[link[0], 2] < kpt_score_thr
                        or kpts_2d[link[1], 2] < kpt_score_thr):
                    # skip the link that should not be drawn
                    continue
                xs_3d = kpts[link_indices, 0]
                ys_3d = kpts[link_indices, 1]
                zs_3d = kpts[link_indices, 2]
                kpt_score = kpts[link_indices, 3]
                if kpt_score.min() > kpt_score_thr:
                    # matplotlib uses RGB color in [0, 1] value range
                    _color = link_color[::-1] / 255.
                    ax.plot(xs_3d, ys_3d, zs_3d, color=_color, zdir='z')

    # convert figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = cv2.cvtColor(img_vis,cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return img_vis

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

def main_video(video_path, video_keypoints):
    
    def onMouse(event, x, y, flags, data):
        if event==1:
            print(img[y][x])

    kp_path=video_keypoints
    keypoints=json.load(open(kp_path))
    video = cv2.VideoCapture(video_path)

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                                [230, 230, 0], [255, 153, 255], [153, 204, 255],
                                [255, 102, 255], [255, 51, 255], [102, 178, 255],
                                [51, 153, 255], [255, 153, 153], [255, 102, 102],
                                [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                [51, 255, 51], [0, 255, 0], [0, 0, 255],
                                [255, 0, 0], [255, 255, 255]])

    skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                            [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]]
    pose_kpt_color = palette[[
        9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]

    
    pose_link_color = palette[[
        0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]
    
    keypoints=np.array(keypoints)
    for items in keypoints:
        for k in items:
            if isinstance(items[k],list):
                items[k]=np.array(items[k])
    cv2.namedWindow("frame",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("frame",onMouse)
    #video.set(cv2.CAP_PROP_POS_FRAMES, Frame_Number)
    for Frame_Numbers in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        _,frame=video.read()
        img=imshow_keypoints_3d([keypoints[Frame_Numbers]],
                            frame,
                            skeleton,
                            pose_kpt_color,
                            pose_link_color,
                            vis_height=800)
        cv2.imshow("frame",img)
        k = cv2.waitKey(1)
        if int(k)==27:
            break

def main_image(image_path, image_keypoints):
    
    def onMouse(event, x, y, flags, data):
        if event==1:
            print(img[y][x])

    kp_path=image_keypoints
    keypoints=json.load(open(kp_path))

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                                [230, 230, 0], [255, 153, 255], [153, 204, 255],
                                [255, 102, 255], [255, 51, 255], [102, 178, 255],
                                [51, 153, 255], [255, 153, 153], [255, 102, 102],
                                [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                [51, 255, 51], [0, 255, 0], [0, 0, 255],
                                [255, 0, 0], [255, 255, 255]])

    skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                            [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]]
    pose_kpt_color = palette[[
        9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]

    
    pose_link_color = palette[[
        0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]
    
    img=cv2.imread(image_path)
    keypoints=np.array(keypoints)
    for items in keypoints:
        for k in items:
            if isinstance(items[k],list):
                items[k]=np.array(items[k])
    
    pose_result_2d = []
    for res in keypoints:
        pose_result_2d.append(res['keypoints'])
    
    img=imshow_keypoints(img,
                         pose_result_2d,
                         skeleton,
                         pose_kpt_color=pose_kpt_color,
                         pose_link_color=pose_link_color)
    
    
    img=imshow_keypoints_3d([keypoints[0]],
                        img,
                        skeleton,
                        pose_kpt_color,
                        pose_link_color,
                        vis_height=800)
                        
    cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    cv2.imshow("frame",img)
    cv2.setMouseCallback("frame",onMouse)
    cv2.waitKey(0)

#main_image(r"test_img\images\4.png", r"vis_results\Keypoints_4.json")
main_video(r"test_vid\3.mp4", r"vis_results\Keypoints_3.json")