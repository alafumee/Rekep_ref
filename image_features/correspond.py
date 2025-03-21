
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from image_features.get_feature import compute_dino_feature

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from magic.match import match
from PIL import Image
from typing import List, Tuple


def match(
        source_img: Image.Image,
        target_img: Image.Image,
        source_center: tuple,
        grasp_center: tuple = None,
        model_size: str = 'base', use_dino_v2: bool = True,
        pca: bool = True, pca_dim: int = 256,
        parameter_save_dir: str = None,
        save_dir: str = 'results/temp',
        top_k: int = 3,
        patch_size=13,
        num_rotation=12,
        use_reflection=False,
        source_object_mask=None,
        target_object_mask=None,
        rotate_fill_color=(0, 0, 0),
        use_recompute=True,
        save_and_show=False,
        only_compute_dino_feature=True,
        # sd_dino=False,
        # dift=False,
):
    """
    Match the source image with the target image with rotation (and reflection) using DINO features and curvature.
    """
    os.makedirs(save_dir, exist_ok=True)
    angle = 360 // num_rotation

    target_imgs = [target_img.rotate(angle * i, fillcolor=rotate_fill_color) for i in range(num_rotation)]
    if use_reflection:
        reflected_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in target_imgs]
        target_imgs = target_imgs + reflected_imgs

    original_imgs = [source_img] + target_imgs

    # if sd_dino and dift:
    #     raise ValueError('Cannot use both sd_dino and dift at the same time')

    # if not sd_dino and not dift:
    result, resized_imgs, downsampled_imgs = compute_dino_feature(source_img, target_imgs, model_size=model_size,
                                                                      use_dino_v2=use_dino_v2, pca=pca, pca_dim=pca_dim)
    # elif sd_dino:
    #     sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100)
    #     result, resized_imgs, downsampled_imgs = compute_sd_dino_feature(source_img, target_imgs, sd_model, sd_aug,
    #                                                                      model_size=model_size, use_dino_v2=use_dino_v2,
    #                                                                      pca=pca, pca_dim=pca_dim)
    # else:
    #     dift_model = SDFeaturizer()
    #     result, resized_imgs, downsampled_imgs = compute_dift_feature(source_img, target_imgs, dift_model, pca=pca,
                                                                    #   pca_dim=pca_dim)

    # if only_compute_dino_feature:
    return result, resized_imgs, downsampled_imgs

def correspond(
    source_img: Image.Image,
    target_img: Image.Image,
    source_xy: tuple[int, int],
    save_dir: str = 'results/temp',
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """
    在target images中找到与source image中指定点对应的点位置

    Args:
        source_img: 源图像
        target_imgs: 目标图像列表
        source_xy: 源图像中的点坐标 (x,y)

    Returns:
        target_points: 目标图像中对应点的坐标列表
        heatmaps: 对应的热力图列表
    """
    os.makedirs(save_dir, exist_ok=True)
    results, resized_imgs, downsampled_images = match(source_img, target_img, source_xy, None, num_rotation=1,
                                                  only_compute_dino_feature=True)
    results = results.permute(0, 2, 3, 1)

    # 计算降采样后的源点坐标
    print("source_img.size: ", source_img.size)
    print("downsampled_images[0].size: ", downsampled_images[0].size)
    print("source_xy: ", source_xy)
    print("resized_imgs[0].size: ", resized_imgs[0].size)
    print("resized_imgs[1].size: ", resized_imgs[1].size)
    downsampled_source_point = (np.array(source_xy) / source_img.size[1] * downsampled_images[0].size[1]).astype(int)
    print("downsampled_source_point: ", downsampled_source_point)
    source_feature = results[0][downsampled_source_point[1], downsampled_source_point[0]]
    resized_source_point = np.array(source_xy) / source_img.size[1] * resized_imgs[0].size[0]


    target_points = (-1, -1)
    heatmaps = []

    # 对每个目标图像计算对应点
    for i in range(results.shape[0]-1):
        print("i th target image: ", i)
        target_feature = results[i+1]
        heatmap = torch.sum(target_feature * source_feature, dim=-1).cpu().numpy()

        # 将热力图缩放到目标图像大小
        heatmap = cv2.resize(heatmap, (resized_imgs[0].size[0], resized_imgs[0].size[1]),
                           interpolation=cv2.INTER_LINEAR)

        # 高斯滤波平滑热力图
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        # print(heatmap.shape)
        # 找到热力图中的最大值位置
        max_y, max_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        target_points=(max_x, max_y)
        heatmaps.append(heatmap)

        # 可视化
        # 创建一个新的图形,设置合适的大小和间距
        plt.figure(figsize=(18, 6))
        plt.subplots_adjust(wspace=0.05)

        # 源图像
        plt.subplot(131)
        plt.imshow(resized_imgs[0])
        plt.scatter(resized_source_point[0], resized_source_point[1], c='r', s=100, marker='*')
        plt.title('源图像', fontsize=12)
        plt.axis('off')

        # 目标图像
        plt.subplot(132)
        plt.imshow(resized_imgs[i+1])
        plt.scatter(max_x, max_y, c='r', s=100, marker='*')
        plt.title('目标图像', fontsize=12)
        plt.axis('off')

        # 热力图
        plt.subplot(133)
        plt.imshow(heatmap, cmap='jet')
        plt.scatter(max_x, max_y, c='r', s=100, marker='*')
        plt.title('对应点热力图', fontsize=12)
        plt.axis('off')
        # 保存
        plt.savefig(os.path.join(save_dir, f'correspond_result.png'))
        # plt.show()
    # print("target_points: ", target_points)
    original_correspond_point = (np.array(target_points) * target_img.size[1] / resized_imgs[1].size[1] ).astype(int)
    print("original_correspond_point: ", original_correspond_point)
    return original_correspond_point, heatmaps

def correspond_batch(
    source_img: Image.Image,
    target_img: Image.Image,
    source_xys: List[Tuple[int, int]],
    save_dir: str = 'results/temp',
) -> Tuple[List[List[Tuple[int, int]]], List[List[np.ndarray]]]:
    """
    在target images中找到与source image中指定点对应的点位置 (批量处理)

    Args:
        source_img: 源图像
        target_imgs: 目标图像列表
        source_xys: 源图像中的点坐标列表 [(x,y), (x,y), ...]

    Returns:
        target_points_batch: 目标图像中对应点的坐标列表的列表
        heatmaps_batch: 对应的热力图列表的列表
    """
    os.makedirs(save_dir, exist_ok=True)
    results, resized_imgs, downsampled_images = match(source_img, target_img, source_xys[0], None, num_rotation=1,
                                                      only_compute_dino_feature=True)
    results = results.permute(0, 2, 3, 1)

    target_points_batch = []
    heatmaps_batch = []

    for source_xy in source_xys:
        # 计算降采样后的源点坐标
        downsampled_source_point = (np.array(source_xy) / source_img.size[1] * downsampled_images[0].size[1]).astype(int)
        source_feature = results[0][downsampled_source_point[1], downsampled_source_point[0]]
        resized_source_point = np.array(source_xy) / source_img.size[1] * resized_imgs[0].size[0]

        target_points = []
        heatmaps = []

        # 对每个目标图像计算对应点
        for i in range(results.shape[0] - 1):
            target_feature = results[i + 1]
            heatmap = torch.sum(target_feature * source_feature, dim=-1).cpu().numpy()

            # 将热力图缩放到目标图像大小
            heatmap = cv2.resize(heatmap, (resized_imgs[0].size[0], resized_imgs[0].size[1]),
                                 interpolation=cv2.INTER_LINEAR)

            # 高斯滤波平滑热力图
            heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

            # 找到热力图中的最大值位置
            max_y, max_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            target_points.append((max_x, max_y))
            heatmaps.append(heatmap)

            # 可视化
            plt.figure(figsize=(18, 6))
            plt.subplots_adjust(wspace=0.05)

            # 源图像
            plt.subplot(131)
            plt.imshow(resized_imgs[0])
            plt.scatter(resized_source_point[0], resized_source_point[1], c='r', s=100, marker='*')
            plt.title('源图像', fontsize=12)
            plt.axis('off')

            # 目标图像
            plt.subplot(132)
            plt.imshow(resized_imgs[i + 1])
            plt.scatter(max_x, max_y, c='r', s=100, marker='*')
            plt.title('目标图像', fontsize=12)
            plt.axis('off')

            # 热力图
            plt.subplot(133)
            plt.imshow(heatmap, cmap='jet')
            plt.scatter(max_x, max_y, c='r', s=100, marker='*')
            plt.title('对应点热力图', fontsize=12)
            plt.axis('off')
            # 保存
            plt.savefig(os.path.join(save_dir, f'correspond_result_{source_xy}.png'))

        original_correspond_points = [(np.array(tp) * target_img.size[1] / resized_imgs[1].size[1]).astype(int) for tp in target_points]
        target_points_batch.append(original_correspond_points)
        heatmaps_batch.append(heatmaps)

    return target_points_batch, heatmaps_batch
