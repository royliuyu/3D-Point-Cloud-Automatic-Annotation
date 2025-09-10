import cherrypy
import os
import json
import subprocess
from jinja2 import Environment, FileSystemLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(BASE_DIR))

import sys
sys.path.append(BASE_DIR)

from tools import check_labels as check, scene_reader
from algos import pre_annotate


class Root(object):
    @cherrypy.expose
    def index(self, scene="", frame=""):
        tmpl = env.get_template('index.html')
        return tmpl.render()

    @cherrypy.expose
    def icon(self):
        tmpl = env.get_template('test_icon.html')
        return tmpl.render()

    @cherrypy.expose
    def ml(self):
        tmpl = env.get_template('test_ml.html')
        return tmpl.render()

    @cherrypy.expose
    def reg(self):
        tmpl = env.get_template('registration_demo.html')
        return tmpl.render()

    @cherrypy.expose
    def view(self, file):
        tmpl = env.get_template('view.html')
        return tmpl.render()

    @cherrypy.expose
    def saveworldlist(self):
        rawbody = cherrypy.request.body.readline().decode('UTF-8')
        data = json.loads(rawbody)

        for d in data:
            scene = d["scene"]
            frame = d["frame"]
            ann = d["annotation"]

            folder_path = os.path.join('data', scene, 'label')
            file_path = os.path.join(folder_path, f'{frame}.json')

            os.makedirs(folder_path, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(ann, f, indent=2, sort_keys=True)

        return "ok"

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def cropscene(self):
        rawbody = cherrypy.request.body.readline().decode('UTF-8')
        data = json.loads(rawbody)

        rawdata = data["rawSceneId"]
        timestamp = rawdata.split("_")[0]

        print("generate scene")
        log_file = os.path.join('temp', f'crop-scene-{timestamp}.log')

        # Ensure temp directory exists
        os.makedirs('temp', exist_ok=True)

        cmd = [
            "python",
            "tools/dataset_preprocess/crop_scene.py",
            "generate",
            f"{rawdata[0:10]}/{timestamp}_preprocessed/dataset_2hz",
            "-",
            data["startTime"],
            data["seconds"],
            data["desc"]
        ]

        try:
            with open(log_file, 'w', encoding='utf-8') as logf:
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,  # Run from project root
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300  # Optional: avoid hanging
                )
            with open(log_file, 'r', encoding='utf-8') as f:
                log = [line.strip() for line in f.readlines()]

            try:
                os.remove(log_file)
            except (OSError, PermissionError) as e:
                log.append(f"[Warning] Could not delete log file: {e}")
        except subprocess.TimeoutExpired:
            log = ["Error: Command timed out."]
            result = subprocess.CompletedProcess(cmd, returncode=1)
        except FileNotFoundError:
            log = [f"Error: Script not found: {' '.join(cmd)}"]
            result = subprocess.CompletedProcess(cmd, returncode=1)
        except Exception as e:
            log = [f"Unexpected error: {str(e)}"]
            result = subprocess.CompletedProcess(cmd, returncode=1)

        return {"code": result.returncode, "log": log}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def checkscene(self, scene):
        scene_path = os.path.join("data", scene)
        ck = check.LabelChecker(scene_path)
        ck.check()
        print(ck.messages)
        return ck.messages

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predict_rotation(self):
        rawbody = cherrypy.request.body.readline().decode('UTF-8')
        data = json.loads(rawbody)
        return {"angle": pre_annotate.predict_yaw(data["points"])}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def auto_annotate(self, scene, frame):
        print("auto annotate ", scene, frame)
        pcd_path = os.path.join('data', scene, 'lidar', f'{frame}.pcd')
        return pre_annotate.annotate_file(pcd_path)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def load_annotation(self, scene, frame):
        res = scene_reader.get_annotations(scene, frame)
        return res

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def load_ego_pose(self, scene, frame):
        ego_pose = scene_reader.read_ego_pose(scene, frame)
        return ego_pose

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def loadworldlist(self):
        rawbody = cherrypy.request.body.readline().decode('UTF-8')
        worldlist = json.loads(rawbody)

        anns = [
            {
                "scene": w["scene"],
                "frame": w["frame"],
                "annotation": scene_reader.get_annotations(w["scene"], w["frame"])
            }
            for w in worldlist
        ]
        return anns

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def datameta(self):
        return scene_reader.get_all_scenes()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def scenemeta(self, scene):
        return scene_reader.get_one_scene(scene)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_all_scene_desc(self):
        scenes_desc = scene_reader.get_all_scene_desc()
        return scenes_desc

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def objs_of_scene(self, scene):
        path = os.path.join("data", scene)
        return self.get_all_objs(path)

    def get_all_objs(self, path):
        label_folder = os.path.join(path, "label")
        if not os.path.isdir(label_folder):
            return []

        files = [f for f in os.listdir(label_folder) if f.lower().endswith('.json')]

        all_objs = {}

        for filename in files:
            file_path = os.path.join(label_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    boxes = json.load(f)
                    for b in boxes:
                        cat = b.get("obj_type")
                        obj_id = b.get("obj_id")
                        if cat is None or obj_id is None:
                            continue
                        k = f"{cat}-{obj_id}"
                        if k in all_objs:
                            all_objs[k]['count'] += 1
                        else:
                            all_objs[k] = {
                                "category": cat,
                                "id": obj_id,
                                "count": 1
                            }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return list(all_objs.values())


if __name__ == '__main__':
    cherrypy.config.update({'log.screen': True})
    cherrypy.quickstart(Root(), '/', config=os.path.join(BASE_DIR, "server.conf"))
else:
    application = cherrypy.Application(Root(), '/', config=os.path.join(BASE_DIR, "server.conf"))