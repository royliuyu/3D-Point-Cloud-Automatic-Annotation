# import os
# import torch
# import torch.nn as nn
# import numpy as np
# # from . import util
# import glob
# import math
# import json
# import sys
# sys.path.append('./algos/models')
# from my_model import get_net, model_configs
#
# # Configure GPU (assuming util.config_gpu handles PyTorch setup)
# # util.config_gpu()
#
# # Constants
# RESAMPLE_NUM = 10
# NUM_POINT = 512
#
#
# state_file = "./algos/models/fpn_resnet_18_epoch_300.pth"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # state_file ='/home/royliu/Documents/datasets/kitti/SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth' ## for kitti dataset ##fpn_resnet_18_epoch_300.pth, model_new_50epoch.pth
# # state_file ='/home/royliu/Documents/datasets/nusc_kitti/check_points/nus_model_new_300epoch.pth'  ## for nusc dataset
# rotation_model = get_net().to(device)
# rotation_model.eval()
# rotation_model.load_state_dict(torch.load(state_file, map_location='cpu'))
#
# # Placeholder for RotationModel (define or convert from Keras model)
# class RotationModel(nn.Module):
#     def __init__(self):
#         super(RotationModel, self).__init__()
#         # Define architecture (replace with actual model architecture)
#         # Example: self.layers = nn.Sequential(...)
#         pass
#
#     def forward(self, x):
#         # Define forward pass
#         # Example: return self.layers(x)
#         pass
#
#
#
# def sample_one_obj(points, num):
#     if points.shape[0] < NUM_POINT:
#         return np.concatenate([points, np.zeros((NUM_POINT - points.shape[0], 3), dtype=np.float32)], axis=0)
#     else:
#         idx = np.arange(points.shape[0])
#         np.random.shuffle(idx)
#         return points[idx[0:num]]
#
# # 模拟 rotation_model.predict() 的输出
# def mock_rotation_model_predict(input_data):
#     batch_size = input_data.shape[0]
#     logits = np.random.randn(batch_size, 3)
#     logits[:, 2] += 10  # 强化第25类的概率
#     return logits
# import numpy as np
#
#
# def predict_yaw(points):
#     # points = np.array(points).reshape((-1, 3))
#     # input_data = np.stack([sample_one_obj(points, NUM_POINT) for _ in range(RESAMPLE_NUM)], axis=0)
#     #
#     # # Convert to PyTorch tensor
#     # input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
#     #
#     # # Predict with model
#     # with torch.no_grad():
#     #     pred_val = rotation_model(input_tensor)
#     #     pred_cls = torch.argmax(pred_val, dim=-1).cpu().numpy()
#     #
#     # print(pred_cls)
#     #
#     # ret = (pred_cls[0] * 3 + 1.5) * np.pi / 180.0
#     # ret = [0, 0, ret] #[0, 0, 4.738568919164605]
#
#     ###########   debuging  ###################
#     points = np.array(points).reshape((-1, 3))
#     input_data = np.stack([
#         sample_one_obj(points, NUM_POINT) for _ in range(9)
#     ], axis=0)
#
#     pred_val = mock_rotation_model_predict(input_data)
#     pred_cls = np.argmax(pred_val, axis=-1)
#     print("Predicted classes:", pred_cls)
#
#     ret = (pred_cls[0] * 3 + 1.5) * np.pi / 180.
#     ret = [0, 0, ret]
#     ###############################
#     print(ret)
#     return ret
#
#
# # Warmup the model
# random_points = np.random.random([1000, 3]).astype(np.float32)
# predict_yaw(random_points)
#
#
# # Placeholder for FilterModel (define or convert from Keras model)
# class FilterModel(nn.Module):
#     def __init__(self):
#         super(FilterModel, self).__init__()
#         # Define architecture
#         pass
#
#     def forward(self, x):
#         # Define forward pass
#         pass
#
#
# use_env = False
# if use_env:
#     filter_model_file = './algos/models/deepannotate_rp_discrimination_env.pth'
# else:
#     filter_model_file = './algos/models/deepannotate_rp_discrimination_obj_xyzi.pth'
#
# # Load filter model
# # filter_model = FilterModel().to(device)
# # filter_model.load_state_dict(torch.load(filter_model_file, map_location=device))
# # filter_model.eval()
# # print(filter_model)
#
#
# def cluster_points(pcdfile):
#     def pre_cluster_pcd(file, output_folder):
#         pre_cluster_exe = "/home/lie/code/pcltest/build/cluster"
#         cmd = "{} {} {}".format(pre_cluster_exe, file, output_folder)
#         os.system(cmd)
#
#     temp_output_folder = "./temppcd"
#     if os.path.exists(temp_output_folder):
#         os.system("rm {}/*".format(temp_output_folder))
#     else:
#         os.mkdir(temp_output_folder)
#
#     pre_cluster_pcd(pcdfile, temp_output_folder)
#
#     obj_files = glob.glob(temp_output_folder + "/*-obj.bin")
#     env_files = glob.glob(temp_output_folder + "/*-env.bin")
#     obj_files.sort()
#     env_files.sort()
#
#     objs = [np.fromfile(c, dtype=np.float32).reshape(-1, 4) for c in obj_files]
#     envs = [np.fromfile(c, dtype=np.float32).reshape(-1, 4) for c in env_files]
#     clusters = list(zip(objs, envs))
#     return clusters
#
#
# def euler_angle_to_rotate_matrix(eu, t):
#     theta = eu
#     # Rotation about x-axis
#     R_x = np.array([
#         [1, 0, 0],
#         [0, math.cos(theta[0]), -math.sin(theta[0])],
#         [0, math.sin(theta[0]), math.cos(theta[0])]
#     ])
#
#     # Rotation about y-axis
#     R_y = np.array([
#         [math.cos(theta[1]), 0, math.sin(theta[1])],
#         [0, 1, 0],
#         [-math.sin(theta[1]), 0, math.cos(theta[1])]
#     ])
#
#     # Rotation about z-axis
#     R_z = np.array([
#         [math.cos(theta[2]), -math.sin(theta[2]), 0],
#         [math.sin(theta[2]), math.cos(theta[2]), 0],
#         [0, 0, 1]
#     ])
#
#     R = np.matmul(R_x, np.matmul(R_y, R_z))
#     t = t.reshape([-1, 1])
#     R = np.concatenate([R, t], axis=-1)
#     R = np.concatenate([R, np.array([0, 0, 0, 1]).reshape([1, -1])], axis=0)
#     return R
#
#
# def sample_one_obj(points, num):
#     centroid = list(np.mean(points[:, :3], axis=0))  # intensity
#     if points.shape[1] > 3:
#         centroid = np.append(centroid, np.zeros(points.shape[1] - 3))
#         centroid = centroid.reshape(1, -1)
#     points = points - centroid
#
#     if points.shape[0] > num:
#         idx = np.arange(points.shape[0])
#         np.random.shuffle(idx)
#         points = points[idx[0:num]]
#     else:
#         sample_idx = np.random.randint(0, high=points.shape[0], size=num - points.shape[0])
#         padding = points[sample_idx]
#         points = np.concatenate([points, padding], axis=0)
#
#     print("input shape", points.shape)
#     return points[:, :3]
#
#
# def filter_nearby_objects(clusters):
#     def nearby(c):
#         center = np.mean(c, axis=0)
#         return np.sum((center * center)[0:2]) < 70 * 70
#
#     ind = [nearby(c[0]) for c in clusters]
#     return np.array(clusters)[ind]
#
#
# def filter_candidate_objects(clusters):
#     objidx = 1 if use_env else 0
#     input_cluster_points = np.stack([sample_one_obj(p[objidx], NUM_POINT) for p in clusters], axis=0)
#
#     # Convert to PyTorch tensor
#     input_tensor = torch.tensor(input_cluster_points, dtype=torch.float32).to(device)
#
#     # Predict with model
#     with torch.no_grad():
#         pred_val = filter_model(input_tensor)
#         prob = torch.softmax(pred_val, dim=1).cpu().numpy()
#
#     pred_cls = prob[:, 1] > 0.5
#     return pred_cls
#
#
# def decide_obj_rotation(objs):
#     input_data = np.stack([sample_one_obj(o, NUM_POINT) for o in objs], axis=0)
#
#     # Convert to PyTorch tensor
#     input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
#
#     # Predict with model
#     with torch.no_grad():
#         pred_val = rotation_model(input_tensor)
#         pred_cls = torch.argmax(pred_val, dim=-1).cpu().numpy()
#
#     ret = (pred_cls * 3 + 1.5) * np.pi / 180.0
#     return ret
#
#
# def calculate_box_dimension(objs, rotation):
#     def calc_one_box(obj, rot):
#         rot = np.array([0, 0, rot])
#         centroid = np.mean(obj, axis=0)
#         obj = obj - centroid
#
#         trans_mat = euler_angle_to_rotate_matrix(rot, np.zeros(3))[:3, :3]
#         relative_position = np.matmul(obj, trans_mat)
#
#         pmin = np.min(relative_position, axis=0)
#         pmax = np.max(relative_position, axis=0)
#         pmin[2] = pmin[2] - 0.2  # Ground correction
#
#         box_dim = pmax - pmin
#         box_center_delta = box_dim / 2 + pmin
#         box_center_delta = np.matmul(trans_mat, box_center_delta)
#         box_center = box_center_delta + centroid
#
#         return np.stack([box_center, box_dim, rot], axis=0)
#
#     return [calc_one_box(obj, rot) for obj, rot in zip(objs, rotation)]
#
#
# def pre_annotate(pcdfile):
#     clusters = cluster_points(pcdfile)
#     clusters = filter_nearby_objects(clusters)
#     cand_ind = filter_candidate_objects(clusters)
#     cand_clusters = np.array(clusters)[cand_ind]
#     cand_clusters = [x[0][:, :3] for x in cand_clusters]
#     cand_rotation = decide_obj_rotation(cand_clusters)
#     boxes = calculate_box_dimension(cand_clusters, cand_rotation)
#     return boxes
#
#
# def translate_np_to_json(boxes):
#     def trans_one_box(box):
#         return {
#             'obj_type': 'Unknown',
#             'psr': {
#                 'position': {
#                     'x': box[0, 0],
#                     'y': box[0, 1],
#                     'z': box[0, 2],
#                 },
#                 'scale': {
#                     'x': box[1, 0],
#                     'y': box[1, 1],
#                     'z': box[1, 2],
#                 },
#                 'rotation': {
#                     'x': box[2, 0],
#                     'y': box[2, 1],
#                     'z': box[2, 2],
#                 }
#             },
#             'obj_id': '',
#         }
#
#     return [trans_one_box(b) for b in boxes]
#
#
# def annotate_file(input, output=None):
#     boxes = pre_annotate(input)
#     boxes_json = translate_np_to_json(boxes)
#
#     if output:
#         with open(output, 'w') as f:
#             json.dump(boxes_json, f)
#
#     return boxes_json
#
# # if __name__ == "__main__":
# #     root_folder = "/home/lie/fast/code/SUSTechPoints-be/data/2020-07-12-15-36-24"
# #     files = os.listdir(root_folder + "/lidar")
# #     files.sort()
# #     for pcdfile in files:
# #         print(pcdfile)
# #         jsonfile = pcdfile.replace(".pcd", ".json")
# #         annotate_file(root_folder + "/lidar/" + pcdfile, root_folder + "/label/" + jsonfile)