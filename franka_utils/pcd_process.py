import argparse
import os
import numpy as np
import zarr
import pickle
from termcolor import cprint

from scipy.spatial.transform import Rotation as R
import fpsample
from sklearn.cluster import DBSCAN
from time import time
import visualizer
from dataclasses import dataclass, field
from typing import List
from utils.realsense_camera import RealSense_Camera


OFFSET=np.array([0.0, 0.0, -0.035])
ROBOT2CAM_POS = np.array([1.2274124573982026, -0.009193338733170697, 0.3683118830445561]) + OFFSET

ROBOT2CAM_QUAT_INITIAL = np.array(
    [0.015873920322366883, -0.18843429010734952, -0.009452363954531973, 0.9819120071477938]
)
OFFSET_ORI_X=R.from_euler('x', -1.2, degrees=True)
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X
OFFSET_ORI_Y=R.from_euler('y', 10, degrees=True)
ori = ori * OFFSET_ORI_Y
OFFSET_ORI_Z=R.from_euler('z', 0, degrees=True)
ori = ori * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()


REALSENSE_SCALE = 0.0002500000118743628

quat = [-0.491, 0.495, -0.505, 0.509]
pos = [0.004, 0.001, 0.014]
# transformation of color link (child) in the robot base frame (parent)
T_link2color = np.concatenate((np.concatenate((R.from_quat(quat).as_matrix(), np.array([pos]).T), axis=1), [[0, 0, 0, 1]]))

##############################################33
T_link2viz = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])


transform_realsense_util = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

@dataclass
class PCDProcConfig:
    random_drop_points: int
    outlier_distance: float
    outlier_count: int
    n_points: int
    work_space: List[List[float]]

# mug_pcd_config = PCDProcConfig(
#     random_drop_points=10000,
#     outlier_distance=0.011,
#     outlier_count=200,
#     n_points=1024,
#     work_space=[
#         [0.16, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.66],  # [-0.66, 0.66]
#         [0.084, 100] # [0.076, 100]
#     ])

# flower_pcd_config = PCDProcConfig(
#     random_drop_points=10000,
#     outlier_distance=0.01,
#     outlier_count=140,
#     n_points=1024,
#     work_space=[
#         [0.16, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.66],  # [-0.66, 0.66]
#         [0.084, 100] # [0.076, 100]
#     ])

# egg_pcd_config = PCDProcConfig(
#     random_drop_points=10000,
#     outlier_distance=0.01,
#     outlier_count=140,
#     n_points=1024,
#     work_space=[
#         [0.16, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.66],  # [-0.66, 0.66]
#         [0.084, 100] # [0.082, 100]
#     ])

# saurce_pcd_config = PCDProcConfig(
#     random_drop_points=15000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.16, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.86],  # [-0.66, 0.66]
#         [0.084, 100] # [0.082, 100]
#     ])

# bear_pcd_config = PCDProcConfig(
#     random_drop_points=15000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.3, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.4],  # [-0.66, 0.66]
#         [0.02, 100] # [0.082, 100]
#     ])

# cube_pcd_config = PCDProcConfig(
#     random_drop_points=10000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.1, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.5],  # [-0.66, 0.66]
#         [0.02, 100] # [0.082, 100]
#     ])

cube_pcd_config = PCDProcConfig(    # jar
    random_drop_points=5000,
    outlier_distance=0.012,
    outlier_count=50,
    n_points=1024,
    work_space=[
        [0.2, 0.8],  # [0.16, 0.8]
        [-0.66, 0.6],  # [-0.66, 0.66]
        [0.005, 0.45] # [0.082, 100]
    ])


# cube_pcd_config = PCDProcConfig(    # drawer
#     random_drop_points=5000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.25, 0.8],  # [0.16, 0.8]
#         [-0.66, 0.6],  # [-0.66, 0.66]
#         [0.005, 0.45] # [0.082, 100]
#     ])

# cube_pcd_config = PCDProcConfig(    # door
#     random_drop_points=5000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.25, 0.8],  # 
#         [-0.66, 0.6],  #
#         [0.005, 0.28] # hack
#     ])

# cube_pcd_config = PCDProcConfig(    # clutter, source
#     random_drop_points=5000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.36, 0.52],
#         [-0.09, 0.12],
#         [0.005, 0.125]
#     ])

# cube_pcd_config = PCDProcConfig(    # clutter, shelf
#     random_drop_points=5000,
#     outlier_distance=0.012,
#     outlier_count=50,
#     n_points=1024,
#     work_space=[
#         [0.36, 0.52],
#         [-0.2, 0.2],
#         [0.3, 0.3 + 0.125]
#     ])


def preprocess_point_cloud(points, cfg=cube_pcd_config, use_cuda=False, debug = False):

    RANDOM_DROP_POINTS = cfg.random_drop_points
    OUTLIER_DISTANCE = cfg.outlier_distance
    OUTLIER_COUNT = cfg.outlier_count
    N_POINTS = cfg.n_points
    WORK_SPACE = cfg.work_space

    time_start = time()

    # ramdom drop points
    # points = points[np.random.choice(points.shape[0], RANDOM_DROP_POINTS, replace=False)]

    def inverse_extrinsic_matrix(extrinsics_matrix):
        extrinsics_matrix_inv = np.linalg.inv(extrinsics_matrix)
        return extrinsics_matrix_inv
    
    if debug:
        time_transform = time()
        cprint(f"Transform time: {time_transform - time_start}", "cyan")
    
    robot2cam_extrinsic_matrix = np.eye(4)
    robot2cam_extrinsic_matrix[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
    robot2cam_extrinsic_matrix[:3, 3] = ROBOT2CAM_POS

    cam_T_robot = inverse_extrinsic_matrix(robot2cam_extrinsic_matrix)
    color_T_cam = inverse_extrinsic_matrix(T_link2color)
    # print(T_link2color)

    # scale
    points_xyz = points[..., :3] * REALSENSE_SCALE
    point_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
    point_homogeneous = T_link2viz @ point_homogeneous.T # TODO: this works! same as in camera_link
    point_homogeneous = robot2cam_extrinsic_matrix @ point_homogeneous
    point_homogeneous = point_homogeneous.T

    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    if debug:
        time_transform = time()
        cprint(f"Transform time: {time_transform - time_start}", "cyan")
        points_viz = points[np.random.choice(points.shape[0], 50000, replace=False)]
        visualizer.visualize_pointcloud(points_viz)
    
    # crop
    # cprint(f"max x {np.max(points[...,0])}, min x {np.min(points[...,0])}, max y {np.min(points[...,1])}, min y {np.min(points[...,1])}", "cyan")
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
        
    # ramdom drop points
    # print(f"points shape after cropping: {points.shape}")
    points = points[np.random.choice(points.shape[0], RANDOM_DROP_POINTS, replace=False)]
    points_xyz = points[..., :3]

    if debug:
        time_drop = time()
        cprint(f"Drop time: {time_drop - time_transform}", "cyan")

        #print(f"points shape after cropping and sampling: {points.shape}")
        # visualizer.visualize_pointcloud(points)

    # DBSCAN clustering TODO: need to be tuned later
    bdscan = DBSCAN(eps=OUTLIER_DISTANCE, min_samples=10)
    labels = bdscan.fit_predict(points_xyz)

    # Then get out of the cluster with less than OUTLIER points or noise
    unique_labels, counts = np.unique(labels, return_counts=True)
    outlier_labels = unique_labels[counts < OUTLIER_COUNT]
    if -1 not in outlier_labels:
        outlier_labels = np.append(outlier_labels, -1)

    points = points[~np.isin(labels, outlier_labels) ]
    points_xyz = points[..., :3]

    if debug:
        print(f"points shape after clustering: {points.shape}")
        # visualizer.visualize_pointcloud(points)

        time_cluster = time()
        cprint(f"Cluster time: {time_cluster - time_drop}", "cyan")

    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, N_POINTS, h=3)
    points = points[sample_indices]

    if debug:
        time_fps = time()
        cprint(f"FPS time: {time_fps - time_cluster}", "cyan")
        print(f"points shape after FPS: {points.shape}")
        # visualizer.visualize_pointcloud(points)

    return points


def pcd_crop(points, cfg=cube_pcd_config, use_cuda=False, debug = False):

    RANDOM_DROP_POINTS = cfg.random_drop_points
    OUTLIER_DISTANCE = cfg.outlier_distance
    OUTLIER_COUNT = cfg.outlier_count
    N_POINTS = cfg.n_points
    WORK_SPACE = cfg.work_space

    time_start = time()

    # ramdom drop points
    # points = points[np.random.choice(points.shape[0], RANDOM_DROP_POINTS, replace=False)]

    def inverse_extrinsic_matrix(extrinsics_matrix):
        extrinsics_matrix_inv = np.linalg.inv(extrinsics_matrix)
        return extrinsics_matrix_inv
    
    if debug:
        time_transform = time()
        cprint(f"Transform time: {time_transform - time_start}", "cyan")
    
    robot2cam_extrinsic_matrix = np.eye(4)
    robot2cam_extrinsic_matrix[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
    robot2cam_extrinsic_matrix[:3, 3] = ROBOT2CAM_POS

    cam_T_robot = inverse_extrinsic_matrix(robot2cam_extrinsic_matrix)
    color_T_cam = inverse_extrinsic_matrix(T_link2color)
    # print(T_link2color)

    # scale
    points_xyz = points[..., :3] * REALSENSE_SCALE
    point_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
    point_homogeneous = T_link2viz @ point_homogeneous.T # TODO: this works! same as in camera_link
    point_homogeneous = robot2cam_extrinsic_matrix @ point_homogeneous
    point_homogeneous = point_homogeneous.T

    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    if debug:
        time_transform = time()
        cprint(f"Transform time: {time_transform - time_start}", "cyan")
        points_viz = points[np.random.choice(points.shape[0], 50000, replace=False)]
    
    
    # visualizer.visualize_pointcloud(points)
    
    # crop
    # cprint(f"max x {np.max(points[...,0])}, min x {np.min(points[...,0])}, max y {np.min(points[...,1])}, min y {np.min(points[...,1])}", "cyan")
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    
    # visualizer.visualize_pointcloud(points)
    
    return points


def pcd_cluster(points, cfg=cube_pcd_config, use_cuda=False, debug = False):
    RANDOM_DROP_POINTS = cfg.random_drop_points
    OUTLIER_DISTANCE = cfg.outlier_distance
    OUTLIER_COUNT = cfg.outlier_count
    N_POINTS = cfg.n_points
    WORK_SPACE = cfg.work_space
    
    # ramdom drop points
    if debug:
        print(f"points shape before cropping: {points.shape}")

    points = points[np.random.choice(points.shape[0], RANDOM_DROP_POINTS, replace=False)]
    points_xyz = points[..., :3]

    if debug:
        time_drop = time()
        # cprint(f"Drop time: {time_drop - time_transform}", "cyan")

        print(f"points shape after cropping and sampling: {points.shape}")
        # visualizer.visualize_pointcloud(points)

    # DBSCAN clustering TODO: need to be tuned later
    bdscan = DBSCAN(eps=OUTLIER_DISTANCE, min_samples=10)
    labels = bdscan.fit_predict(points_xyz)

    # Then get out of the cluster with less than OUTLIER points or noise
    unique_labels, counts = np.unique(labels, return_counts=True)
    outlier_labels = unique_labels[counts < OUTLIER_COUNT]
    if -1 not in outlier_labels:
        outlier_labels = np.append(outlier_labels, -1)

    points = points[~np.isin(labels, outlier_labels) ]
    points_xyz = points[..., :3]

    if debug:
        print(f"points shape after clustering: {points.shape}")
        # visualizer.visualize_pointcloud(points)

        time_cluster = time()
        cprint(f"Cluster time: {time_cluster - time_drop}", "cyan")

    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, N_POINTS, h=3)
    points = points[sample_indices]

    if debug:
        time_fps = time()
        cprint(f"FPS time: {time_fps - time_cluster}", "cyan")
        print(f"points shape after FPS: {points.shape}")
        
    # visualizer.visualize_pointcloud(points)

    return points


if __name__ == "__main__":
    id = "f0211830"
    realsense_camera = RealSense_Camera(type="L515", id=id)
    realsense_camera.prepare()
    point_cloud, rgbd_frame = realsense_camera.get_frame()

    preprocess_point_cloud(points=point_cloud, debug=False)
    # save to npy
    np.save("./data/moon.npy", point_cloud)