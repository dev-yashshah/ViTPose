import numpy as np
from matplotlib import pyplot as plt
import cv2

class kpt_visualization:
    def __init__(self) -> None:
        self.palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                                [230, 230, 0], [255, 153, 255], [153, 204, 255],
                                [255, 102, 255], [255, 51, 255], [102, 178, 255],
                                [51, 153, 255], [255, 153, 153], [255, 102, 102],
                                [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                [51, 255, 51], [0, 255, 0], [0, 0, 255],
                                [255, 0, 0], [255, 255, 255]])

        self.skeleton = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                                [7, 8], [8, 9], [8, 10], [10, 11], [11, 12], [8, 13],
                                [13, 14], [14, 15]])
        
        self.pose_kpt_color = np.array(self.palette[[
            9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]])

        
        self.pose_link_color = np.array(self.palette[[
            0, 0, 0, 16, 16, 16, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]])

    def imshow_keypoints_3d(
            self,
            pose_result,
            keypoint_label = False,
            kpt_score_thr=0.3,
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

        #img_h, img_w = img_data["res_h"], img_data["res_w"]
        img_h, img_w = 400, 400
        plt.ioff()
        fig = plt.figure(figsize=(img_w * 0.01, img_h * 0.01))

        
        kpts = pose_result
        
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        valid = kpts[:, 3] >= kpt_score_thr
        
        ax_idx = 1
        ax = fig.add_subplot(1, 1, ax_idx, projection='3d')
        ax.view_init(
            elev=axis_elev,
            azim=axis_azimuth
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
        
        if self.pose_kpt_color is not None:
            assert len(self.pose_kpt_color) == len(kpts)
            x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)
            # matplotlib uses RGB color in [0, 1] value range
            _color = self.pose_kpt_color[..., ::-1] / 255.
            ax.scatter(
                x_3d[valid],
                y_3d[valid],
                z_3d[valid],
                marker='o',
                color=_color[valid],
            )
        if keypoint_label:
            for pnt_cnt, points in enumerate(zip(x_3d[valid],y_3d[valid], z_3d[valid])):
                ax.text(points[0][0] * (1 + 0.01), 
                        points[1][0] * (1 + 0.01), 
                        points[2][0] * (1 + 0.01), 
                        s=str(pnt_cnt), 
                        fontsize=12)
            
        if self.skeleton is not None and self.pose_link_color is not None:
            assert len(self.pose_link_color) == len(self.skeleton)
            
            for link, link_color in zip(self.skeleton, self.pose_link_color):
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

def main():
    test_kpt = np.array([[-0.0640245 ,  0.23428799 , 0.936804  ],
                        [-0.19434163 , 0.20985073  ,0.94658124],
                        [-0.17516214 , 0.24163711  ,0.50524527],
                        [-0.18336831 , 0.30824962  ,0.05602492],
                        [ 0.06629286 , 0.25872532  ,0.9270268 ],
                        [ 0.03384836 , 0.27251428  ,0.48553762],
                        [ 0.00785348 , 0.3288635   ,0.03559024],
                        [-0.07981335 , 0.26691678  ,1.1674438 ],
                        [-0.07980996 , 0.2684659   ,1.4245167 ],
                        [-0.09750853 , 0.2853243   ,1.6068532 ],
                        [ 0.05091807 , 0.28973007  ,1.3519264 ],
                        [ 0.11153924 , 0.34123427  ,1.0846288 ],
                        [ 0.1643369  , 0.39309895  ,0.8440208 ],
                        [-0.20954525 , 0.2625452   ,1.3474181 ],
                        [-0.27717015 , 0.29909205  ,1.0793276 ],
                        [-0.33582887 , 0.32183474  ,0.8355875 ]])
    
    vis = kpt_visualization()
    img=vis.imshow_keypoints_3d(test_kpt)
    cv2.imshow("m",img)
    cv2.waitKey(0)
#main()