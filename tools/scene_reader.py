
import os
import json
import numpy as np
import torch
from algos.models.my_model import get_net
from algos.util import pointCloud2bev, filter_lidar, decode, post_process
from algos.util import bev2lidar, camera2lidar, R_y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_net().to(device)
model_weight_path = os.path.join(os.path.dirname(__file__), '../algos', 'models', 'fpn_resnet_18_epoch_300.pth')

if os.path.isfile(model_weight_path):
    print(f"Loading trained model from {model_weight_path}")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
else:
    raise FileNotFoundError(f"Model file not found: {model_weight_path}")



def load_lidar(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(this_dir, "../data")
PI = 3.141592653589793

def get_all_scenes():
    all_scenes = get_scene_names()
    print(all_scenes)
    return list(map(get_one_scene, all_scenes))

def get_all_scene_desc():
    names = get_scene_names()
    descs = {}
    for n in names:
        descs[n] = get_scene_desc(n)
    return descs

def get_scene_names():
      scenes = os.listdir(root_dir)
      scenes = filter(lambda s: not os.path.exists(os.path.join(root_dir, s, "disable")), scenes)
      scenes = list(scenes)
      scenes.sort()
      return scenes

def get_scene_desc(s):
    scene_dir = os.path.join(root_dir, s)
    if os.path.exists(os.path.join(scene_dir, "desc.json")):
        with open(os.path.join(scene_dir, "desc.json")) as f:
            desc = json.load(f)
            return desc
    return None

### for Kittie dataset
def get_one_scene(s):
    scene = {
        "scene": s,
        "frames": []
    }

    scene_dir = os.path.join(root_dir, s)

    frames = os.listdir(os.path.join(scene_dir, "lidar"))

    frames.sort()

    scene["lidar_ext"]="pcd"
    for f in frames:
        #if os.path.isfile("./data/"+s+"/lidar/"+f):
        filename, fileext = os.path.splitext(f)
        scene["frames"].append(filename)
        scene["lidar_ext"] = fileext


    if os.path.exists(os.path.join(scene_dir, "desc.json")):
        with open(os.path.join(scene_dir, "desc.json")) as f:
            desc = json.load(f)
            scene["desc"] = desc

    calib = {}
    calib_camera={}
    calib_radar={}
    calib_aux_lidar = {}
    if os.path.exists(os.path.join(scene_dir, "calib")):

        calib_path = os.path.join(scene_dir, "calib")
        if os.path.exists(calib_path):
            calib_files = [f for f in os.listdir(calib_path) if f.endswith(".txt")]
            if calib_files:
                first_calib_file = os.path.join(calib_path, calib_files[0])
                with open(first_calib_file, 'r') as f:
                    calib_data = {}
                    for line in f.readlines():
                        key_value = line.strip().split(' ')
                        key = key_value[0].rstrip(':')
                        values = list(map(float, key_value[1:]))
                        calib_data[key] = values


                P2 = calib_data.get("P2")
                R0_rect = calib_data.get("R0_rect")  # 3x3 rectify rotation
                Tr_velo = calib_data.get("Tr_velo_to_cam")

                if P2 and R0_rect and Tr_velo:

                    K = [
                        P2[0], P2[1], P2[2],
                        P2[4], P2[5], P2[6],
                        P2[8], P2[9], P2[10]
                    ]

                    R_lidar_to_cam = Tr_velo[:12]
                    T_lidar_to_cam = [
                        R_lidar_to_cam[0], R_lidar_to_cam[1], R_lidar_to_cam[2], R_lidar_to_cam[3],
                        R_lidar_to_cam[4], R_lidar_to_cam[5], R_lidar_to_cam[6], R_lidar_to_cam[7],
                        R_lidar_to_cam[8], R_lidar_to_cam[9], R_lidar_to_cam[10], R_lidar_to_cam[11],
                        0.0, 0.0, 0.0, 1.0
                    ]

                    calib_camera["front"] = {
                        "intrinsic": K,
                        "extrinsic": T_lidar_to_cam  # 或者根据需求结合 R0_rect
                    }


        if os.path.exists(os.path.join(scene_dir, "calib", "radar")):
            calibs = os.listdir(os.path.join(scene_dir, "calib", "radar"))
            for c in calibs:
                calib_file = os.path.join(scene_dir, "calib", "radar", c)
                calib_name, _ = os.path.splitext(c)
                if os.path.isfile(calib_file):
                    #print(calib_file)
                    with open(calib_file)  as f:
                        cal = json.load(f)
                        calib_radar[calib_name] = cal
        if os.path.exists(os.path.join(scene_dir, "calib", "aux_lidar")):
            calibs = os.listdir(os.path.join(scene_dir, "calib", "aux_lidar"))
            for c in calibs:
                calib_file = os.path.join(scene_dir, "calib", "aux_lidar", c)
                calib_name, _ = os.path.splitext(c)
                if os.path.isfile(calib_file):
                    #print(calib_file)
                    with open(calib_file)  as f:
                        cal = json.load(f)
                        calib_aux_lidar[calib_name] = cal

    # camera names
    camera = []
    camera_ext = ""
    cam_path = os.path.join(scene_dir, "camera")
    if os.path.exists(cam_path):
        cams = os.listdir(cam_path)
        for c in cams:
            cam_file = os.path.join(scene_dir, "camera", c)
            if os.path.isdir(cam_file):
                camera.append(c)

                if camera_ext == "":
                    #detect camera file ext
                    files = os.listdir(cam_file)
                    if len(files)>=2:
                        _,camera_ext = os.path.splitext(files[0])

    if camera_ext == "":
        camera_ext = ".jpg"
    scene["camera_ext"] = camera_ext

    radar = []
    radar_ext = ""
    radar_path = os.path.join(scene_dir, "radar")
    if os.path.exists(radar_path):
        radars = os.listdir(radar_path)
        for r in radars:
            radar_file = os.path.join(scene_dir, "radar", r)
            if os.path.isdir(radar_file):
                radar.append(r)
                if radar_ext == "":
                    #detect camera file ext
                    files = os.listdir(radar_file)
                    if len(files)>=2:
                        _,radar_ext = os.path.splitext(files[0])

    if radar_ext == "":
        radar_ext = ".pcd"
    scene["radar_ext"] = radar_ext


    # aux lidar names
    aux_lidar = []
    aux_lidar_ext = ""
    aux_lidar_path = os.path.join(scene_dir, "aux_lidar")
    if os.path.exists(aux_lidar_path):
        lidars = os.listdir(aux_lidar_path)
        for r in lidars:
            lidar_file = os.path.join(scene_dir, "aux_lidar", r)
            if os.path.isdir(lidar_file):
                aux_lidar.append(r)
                if radar_ext == "":
                    #detect camera file ext
                    files = os.listdir(radar_file)
                    if len(files)>=2:
                        _,aux_lidar_ext = os.path.splitext(files[0])

    if aux_lidar_ext == "":
        aux_lidar_ext = ".pcd"
    scene["aux_lidar_ext"] = aux_lidar_ext



    if  True: #not os.path.isdir(os.path.join(scene_dir, "bbox.xyz")):
        scene["boxtype"] = "psr"

        if camera:
            scene["camera"] = camera
        if radar:
            scene["radar"] = radar
        if aux_lidar:
            scene["aux_lidar"] = aux_lidar
        if calib_camera:
            calib["camera"] = calib_camera
        if calib_radar:
            calib["radar"] = calib_radar
        if calib_aux_lidar:
            calib["aux_lidar"] = calib_aux_lidar

    scene["calib"] = calib
    return scene

### read label

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    def parse_mat(line):
        return np.array([float(x) for x in line.split(' ')[1:]]).reshape(3, 4)

    for line in lines:
        if 'Tr_velo_to_cam' in line:
            Tr_velo_to_cam = parse_mat(line)
        elif 'R0_rect' in line:
            R0_rect = np.eye(4)
            R0_rect[:3, :3] = np.array([float(x) for x in line.split(' ')[1:]]).reshape(3, 3)

    Tr_velo_to_cam_ext = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam_ext)

    return {
        'Tr_velo_to_cam': Tr_velo_to_cam,
        'R0_rect': R0_rect,
        'Tr_cam_to_velo': Tr_cam_to_velo
    }


def get_annotations(scene, frame):

    filename = os.path.join(root_dir, scene, "label", frame + ".txt")
    calib_file = os.path.join(root_dir, scene, "calib", frame + ".txt")
    lidar_file = os.path.join(root_dir, scene, "lidar", frame + ".bin")

    if not os.path.isfile(calib_file):
        print(f"Missing calibration file: {calib_file}")
        return []

    if os.path.isfile(filename):
        return label2annotation(filename, calib_file)

    print(f"Label file is not existed to loaded: {filename}. Running prediction instead ...")

    try:
        lidar = load_lidar(lidar_file)
    except Exception as e:
        print(f"Error loading LiDAR file: {e}")
        return []

    lidar = filter_lidar(lidar)
    bev_img = pointCloud2bev(lidar)
    bev_tensor = torch.tensor(bev_img).float().unsqueeze(0).to(device)  # 添加批次维度

    with torch.no_grad():
        preds = model(bev_tensor)
    res = decode(preds).cpu().numpy().astype(np.float32)  ## get BEV pixel prediction
    res = post_process(res)[0]  ## x, y, z, h, w, l, yaw. get BEV map scale (608,608)
    res = bev2lidar(res) ## get x, y, z, h, w, l, yaw, xyz with physical scale

    annotations = prediction2annotation(res, calib_file)

    return annotations

def prediction2annotation(predictions: dict, calib_file):
    """
    Convert predictions to annotation format with corrected ry.

    Assumes:
        - predictions use KITTI-style ry (camera Y-axis rotation)
        - but positions may be in lidar coordinates
        - we still need to correct ry's semantic meaning
    """
    calib = read_calib(calib_file)
    Tr_cam_to_velo = calib['Tr_cam_to_velo']  # 4x4
    R_cam_to_velo = Tr_cam_to_velo[:3, :3]  # 3x3 rotation

    class_id_to_type = {
        0: "Pedestrian",
        1: "Car",
        2: "Cyclist"
    }

    annotations = []
    for class_id, preds in predictions.items():
        if len(preds) == 0:
            continue

        obj_type = class_id_to_type.get(class_id, f"Class{class_id}")

        for idx, bbox in enumerate(preds):
            score, x_lidar, y_lidar, z_lidar, h, w, l, ry_cam = bbox
            x_lidar, y_lidar, z_lidar = float(x_lidar), float(y_lidar), float(z_lidar)
            h, w, l, ry_cam = float(h), float(w), float(l), float(ry_cam)

            forward_in_cam = R_y(ry_cam) @ np.array([0, 0, 1])  # [cos, 0, sin]
            forward_in_lidar = R_cam_to_velo @ forward_in_cam
            forward_in_lidar = forward_in_lidar[:3]
            forward_in_lidar /= np.linalg.norm(forward_in_lidar)

            ry_lidar = np.arctan2(forward_in_lidar[1], forward_in_lidar[0])
            annotations.append({
                'obj_id': f"{class_id}_{idx}",
                'obj_type': obj_type,
                'psr': {
                    'position': {
                        'x': x_lidar,
                        'y': y_lidar,
                        'z': z_lidar
                    },
                    'rotation': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': ry_lidar
                    },
                    'scale': {
                        'x': w,
                        'y': l,
                        'z': h
                    }
                },
                'score': float(score)
            })

    return annotations

def label2annotation(label_file, calib_file):
    """Parse KITTI label file and return list of annotation dicts"""
    calib = read_calib(calib_file)
    Tr_cam_to_velo = calib['Tr_cam_to_velo']  # 4x4 matrix

    R_cam_to_velo = Tr_cam_to_velo[:3, :3]  #

    annotations = []
    with open(label_file, 'r') as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts or parts[0] == 'DontCare':
                continue

            try:
                obj_type = parts[0]
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x_cam = float(parts[11])
                y_cam = float(parts[12])
                z_cam = float(parts[13])
                ry_cam = float(parts[14])
                y_bottom_cam = y_cam - h / 2.0

                center_cam = np.array([x_cam, y_bottom_cam, z_cam])
                center_lidar = camera2lidar(center_cam, Tr_cam_to_velo)

                forward_in_cam = R_y(ry_cam) @ np.array([1, 0, 0])  # [cos(ry), 0, sin(ry)]

                forward_in_lidar = R_cam_to_velo @ forward_in_cam
                forward_in_lidar = forward_in_lidar[:3]
                forward_in_lidar /= np.linalg.norm(forward_in_lidar)

                ry_lidar = np.arctan2(forward_in_lidar[1], forward_in_lidar[0])

                annotations.append({
                    'obj_id': str(idx),
                    'obj_type': obj_type,
                    'psr': {
                        'position': {'x': center_lidar[0], 'y': center_lidar[1], 'z': center_lidar[2]},
                        'rotation': {'x': 0.0, 'y': 0.0, 'z': ry_lidar},  # ✅ 使用修正后的 ry
                        'scale': {'x': l, 'y': w, 'z': h}
                    }
                })

            except Exception as e:
                print(f"Error parsing line {idx}: {e}")

    return annotations


def read_ego_pose(scene, frame):
    filename = os.path.join(root_dir, scene, "ego_pose", frame+".json")
    if (os.path.isfile(filename)):
      with open(filename,"r") as f:
        p=json.load(f)
        return p
    else:
      return None

def save_annotations(scene, frame, anno):
    filename = os.path.join(root_dir, scene, "label", frame+".json")
    with open(filename, 'w') as outfile:
            json.dump(anno, outfile)

if __name__ == "__main__":
    print(get_all_scenes())