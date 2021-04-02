from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg

import matplotlib.pyplot as plt
import random, cv2, time, os, shutil, ownlabelme2COCO

import torch
print(torch.__version__)

path = "/home/julius/Schreibtisch/test_dir/"

dataset_path = path + "1_Datensaetze/data100/"
train_set_path = dataset_path + "train_split/"
test_set_path = dataset_path + "test_split/"

starttime = time.strftime("%d,%m,%Y-%H,%M")
model_path = path + "trained_models/detectron2/{}/".format(starttime)
evaluation_path = path + "model_evaluation/detectron2/{}/".format(starttime)

class uneven_list_error(Exception):
    # raised, when the two lists which are needed to seperate the data are uneven.
    pass

try:
    if not os.path.isdir(train_set_path) & os.path.isdir(test_set_path):
        os.mkdir(train_set_path)
        os.mkdir(test_set_path)
    
        images = sorted([element for element in os.listdir(dataset_path) if element.lower().endswith(".jpg")])
        jsons = sorted([element for element in os.listdir(dataset_path) if element.endswith(".json")])

        if len(images) != len(jsons):
            raise uneven_list_error

        for count in range(len(images)):
            if count % 5 == 0:
                shutil.move(dataset_path + images[count], test_set_path + images[count])
                shutil.move(dataset_path + jsons[count], test_set_path + jsons[count])
            else:
                shutil.move(dataset_path + images[count], train_set_path + images[count])
                shutil.move(dataset_path + jsons[count], train_set_path + jsons[count])
        
        ownlabelme2COCO.main(test_set_path)
        ownlabelme2COCO.main(train_set_path)

except uneven_list_error:
    print("[ERROR] List lengths don't match! There are {} Images and {} json-Files. Please check directory!".format(len(images), len(jsons)))

load_coco_json(train_set_path + "COCO_json/output.json", train_set_path, "train_set")
register_coco_instances("train_set", {}, train_set_path + "COCO_json/output.json", train_set_path)
train_set_metadata = MetadataCatalog.get("train_set")
train_set_data = DatasetCatalog.get("train_set")

load_coco_json(test_set_path + "COCO_json/output.json", test_set_path, "test_set")
register_coco_instances("test_set", {}, test_set_path + "COCO_json/output.json", test_set_path)
test_set_metadata = MetadataCatalog.get("test_set")
test_set_data = DatasetCatalog.get("test_set")

random_image = random.sample(train_set_data, 1)[0]
image = cv2.imread(random_image["file_name"])
visualizer = Visualizer(image, metadata=train_set_metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
visualization = visualizer.draw_dataset_dict(random_image)
plt.figure(figsize=(25, 25))
plt.imshow(visualization.get_image()[:,:, ::-1])
plt.axis("off")

print(train_set_metadata)

config = get_cfg()
config.merge_from_file("/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
config.DATASETS.TRAIN = ("train_set",)
config.DATASETS.TEST = ("test_set",)
config.DATALOADER.NUM_WORKERS = 2
config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
config.OUTPUT_DIR = model_path
config.SOLVER.IMS_PER_BATCH = 16
config.SOLVER.REFERENCE_WORLD_SIZE = 1
config.SOLVER.BASE_LR = 0.00025
config.SOLVER.MAX_ITER = 300
config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2
config.MODEL.ROI_HEADS.NUM_CLASSES = 11
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
config.NUM_GPUS = 0

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()
trainer = DefaultTrainer(config)
trainer.train()