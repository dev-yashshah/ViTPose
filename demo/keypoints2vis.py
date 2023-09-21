import json
import numpy as np
from matplotlib import pyplot as plt
import cv2

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
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        
        valid = kpts[:, 3] >= kpt_score_thr

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

        if not dummy and skeleton is not None and pose_link_color is not None:
            pose_link_color = np.array(pose_link_color)
            assert len(pose_link_color) == len(skeleton)
            for link, link_color in zip(skeleton, pose_link_color):
                link_indices = [_i for _i in link]
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

def main(Frame_Number=0):
    
    def onMouse(event, x, y, flags, data):
        if event==1:
            print(img[y][x])

    kp_path=r"vis_results\Keypoints_1.json"
    keypoints=json.load(open(kp_path))
    video = cv2.VideoCapture(r"test_vid\1.mp4")

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

    pose_kpt_color = palette[list(range(17))]

    pose_link_color = palette[[
        0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
    ]]
    pose_link_color = palette[list(range(16))]
    keypoints=np.array(keypoints)
    for items in keypoints:
        for k in items:
            if isinstance(items[k],list):
                items[k]=np.array(items[k])

    video.set(cv2.CAP_PROP_POS_FRAMES, Frame_Number)
    _,frame=video.read()
    img=imshow_keypoints_3d([keypoints[Frame_Number]],
                        frame,
                        skeleton,
                        pose_kpt_color,
                        pose_link_color,
                        vis_height=800)
    cv2.namedWindow("frame",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("frame",img)
    cv2.setMouseCallback("frame",onMouse)
    cv2.waitKey(0)

main()