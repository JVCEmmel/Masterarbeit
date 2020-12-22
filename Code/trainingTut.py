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

# my imports
import helpfulFunctions

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/data100/"


# load and register the dataset
load_coco_json(path + "output.json", path[:-1], "test_set")
register_coco_instances("test_set", {}, path + "output.json", path[:-1])

# get the Metadata
test_set_metadata = MetadataCatalog.get("test_set")
print(test_set_metadata)

dataset_dicts = DatasetCatalog.get("test_set")

# display an image of the dataset

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:,:,::-1], metadata=test_set_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    
    helpfulFunctions.showImage(vis.get_image()[:,:,::-1])

# actual training
config = get_cfg()
config.merge_from_file(
    "/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

config.DATASETS.TRAIN = ("train_set",)
config.DATASETS.TEST = ("test_set",)
config.DATALOADER.NUM_WORKERS = 2
config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
config.OUTPUT_DIR = "/home/julius/PowerFolders/Masterarbeit/detectron_training_output/"
config.SOLVER.IMS_PER_BATCH = 2
config.SOLVER.BASE_LR = 0.02
config.SOLVER.MAX_ITER = (300)
config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
config.MODEL.ROI_HEADS.NUM_CLASSES = 16

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(config)
trainer.resume_or_load(resume=False)
trainer.train()

config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(config)

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata=test_set_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    helpfulFunctions.showImage(v.get_image()[:,:,::-1])
