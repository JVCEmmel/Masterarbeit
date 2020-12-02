# dataset imports
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import random
import cv2

# training imports
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"

# load and register the dataset
load_coco_json(path + "output.json", path[:-1], "first_anno_ds")
register_coco_instances("first_anno_ds", {}, path + "output.json", path[:-1])

# get the Metadata
first_anno_ds_metadata = MetadataCatalog.get("first_anno_ds")
print(first_anno_ds_metadata)

dataset_dicts = DatasetCatalog.get("first_anno_ds")

# display an image of the dataset
"""
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:,:,::-1], metadata=first_anno_ds_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    
    cv2.imshow("Bild", vis.get_image()[:,:,::-1])
    cv2.waitKey(3000)
    cv2.destroyWindow("Bild")
"""

# actual training
config = get_cfg()
config.merge_from_file(
    "/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

config.DATASETS.TRAIN = ("first_anno_ds",)
config.DATASETS.TEST = ()
config.DATALOADER.NUM_WORKERS = 2
config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
config.SOLVER.IMS_PER_BATCH = 2
config.SOLVER.BASE_LR = 0.02
config.SOLVER.MAX_ITER = (300)
config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
config.MODEL.ROI_HEADS.NUM_CLASSES = 16

"""
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(config)
trainer.resume_or_load(resume=False)
trainer.train()
"""

config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
config.DATASETS.TEST = ("first_anno_ds",)
predictor = DefaultPredictor(config)

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata=first_anno_ds_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Bild", v.get_image()[:,:,::-1])
    cv2.waitKey(30000)
    cv2.destroyWindow("Bild")