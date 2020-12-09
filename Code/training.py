from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
import cv2 as cv
import os
import random
import helpfulFunctions, ownlabelme2COCO, predictor

def showRandomImage(train_set_data, train_set_metadata):
    random_image = random.sample(train_set_data, 1)[0]
    image = cv.imread(random_image["file_name"])
    visualizer = Visualizer(image, metadata=train_set_metadata, scale=0.33, instance_mode=ColorMode.SEGMENTATION)
    visualization = visualizer.draw_dataset_dict(random_image)

    helpfulFunctions.showImage(visualization.get_image())

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/data100/"
train_path = path + "train_split/"
test_path = path + "test_split/"
dataset_name = path.split("/")[-2]

if os.path.isdir(train_path) == False:
    helpfulFunctions.splitTrainTest(path, 5)
    ownlabelme2COCO.main(train_path)
    ownlabelme2COCO.main(test_path)

#load and register dataset, generate Metadata - all required
load_coco_json(train_path + "output.json", train_path[:-1], "train_set")
register_coco_instances("train_set", {}, train_path + "output.json", train_path[:-1])
train_set_metadata = MetadataCatalog.get("train_set")
train_set_data = DatasetCatalog.get("train_set")

load_coco_json(test_path + "output.json", test_path[:-1], "test_set")
register_coco_instances("test_set", {}, test_path + "output.json", test_path[:-1])
test_set_metadata = MetadataCatalog.get("test_set")
test_set_data = DatasetCatalog.get("test_set")

print(train_set_metadata)
print(test_set_metadata)

showRandomImage(train_set_data, train_set_metadata)

config = get_cfg()

config.merge_from_file("/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# customize the config
config.DATASETS.TRAIN = ("train_set",)
config.DATASETS.TEST = ("test_set",)
config.DATALOADER.NUM_WORKERS = 2
config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
config.OUTPUT_DIR = "/home/julius/PowerFolders/Masterarbeit/detectron_training_output/" + dataset_name + "/"
config.SOLVER.IMS_PER_BATCH = 2
config.SOLVER.REFERENCE_WORLD_SIZE = 1
config.SOLVER.BASE_LR = 0.02
config.SOLVER.MAX_ITER = (300)
config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
config.MODEL.ROI_HEADS.NUM_CLASSES = 16
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

print(config)
if os.path.isdir(path+"evaluation") == False:
        os.makedirs(path+"evaluation")
evaluator = COCOEvaluator("test_set", config, False, output_dir=path+"evaluation/")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(config)
trainer.resume_or_load(resume=False)
trainer.train()
trainer.test(config, trainer.model, evaluators=evaluator)

predictor.serialPredictor(config, path, test_set_data, test_set_metadata)
