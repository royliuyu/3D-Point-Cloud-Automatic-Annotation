# import tensorflow as tf
#
#
#
# def config_gpu():
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(e)

import numpy as np
import torch
import torch.nn.functional as F

num_classes = 3
down_ratio=4
peak_thresh=0.2

boundary = {  ## kittie's
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

minX = boundary['minX']
maxX = boundary['maxX']
minY = boundary['minY']
maxY = boundary['maxY']
minZ = boundary['minZ']
maxZ = boundary['maxZ']

BEV_WIDTH = 608  # across y axis -25m ~ 25m
BEV_HEIGHT = 608  # across x axis 0m ~ 50m
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT


def filter_lidar(lidar):
    minX, maxX = boundary['minX'], boundary['maxX']
    minY, maxY = boundary['minY'], boundary['maxY']
    minZ, maxZ = boundary['minZ'], boundary['maxZ']

    # mask = (
    #     (lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
    #     (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
    #     (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ)
    # )

    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

    lidar = lidar[mask]
    lidar[:, 2] -= minZ  # Normalize Z
    return lidar


def pointCloud2bev(PointCloud):
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map

    return RGB_Map


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def decode(preds, K=40):
    hm_cen, cen_offset, direction, z_coor, dim = preds.values()
    batch_size, num_classes, height, width = hm_cen.size()
    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()  ## (batch_size, 40, 1)
    scores = scores.view(batch_size, K, 1)
    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections

def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])

def post_process(detections, num_classes=num_classes, down_ratio=down_ratio, peak_thresh=peak_thresh):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return: BEV
    """

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],  ## score
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / bound_size_y * BEV_WIDTH,
                detections[i, inds, 6:7] / bound_size_x * BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]
        ret.append(top_preds)

    return ret


def bev2lidar(bev_predictions):
    """
    Convert BEV-predicted bounding boxes to lidar coordinates,

    :param bev_predictions: dict, {class_id: np.ndarray([score, x_bev, y_bev, z, h, w, l, ry])}
    :return: dict, {class_id: np.ndarray([score, x_cam, y_cam, z, h, w, l, ry])}
    """
    physical_predictions = {}

    for class_id, bboxes in bev_predictions.items():
        if len(bboxes) == 0:
            physical_predictions[class_id] = bboxes
            continue

        # Unpack each bbox: [score, x_bev, y_bev, z, h, w, l, ry]
        score, x_bev, y_bev, z, h, w, l, ry = np.split(bboxes, 8, axis=1)

        # Convert BEV x, y to lidar coordinates
        y_phy = (x_bev / BEV_HEIGHT) * bound_size_y + minY
        x_phy = (y_bev / BEV_WIDTH) * bound_size_x + minX
        z += minZ +0.75 ###
        w = w/BEV_WIDTH * bound_size_x
        l = l/BEV_HEIGHT * bound_size_y

        # Rebuild the bbox with physical coordinates
        physical_bboxes = np.concatenate([
            score,     # score
            x_phy ,  #
            y_phy ,
            z,  # z remains unchanged
            h,  # height
            l,  # width
            w,  # length
            ry         # rotation
        ], axis=1)

        physical_predictions[class_id] = physical_bboxes

    return physical_predictions



def camera2lidar(camera_point, Tr_cam_to_velo):
    """
    将一个相机坐标系下的点转换为激光雷达坐标系下的点
    camera_point: shape=(3,) -> [x, y, z]
    """
    point = np.array([*camera_point, 1])  # 齐次坐标
    lidar_point = np.dot(Tr_cam_to_velo, point)
    return lidar_point[:3]

''' 修正的转换：不是直接传递 ry， Tr_cam_to_velo 不仅包含平移，还包含旋转。这个旋转会影响 ry 的方向语义'''
# 辅助函数：绕 Y 轴旋转矩阵
def R_y(ry):
    return np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])