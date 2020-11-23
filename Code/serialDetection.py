#pytorch, torchvision, detectron2
import torch, torchvision
assert torch.__version__.startswith("1.6")
import detectron2

#detectron logger
from detectron2.utils.logger import setup_logger
setup_logger()

#additional libraries
import numpy as np
import os, json, cv2, random

#detectron utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

try:
    path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"
    
    image_list = sorted([picture for picture in os.listdir(path) if picture.endswith(".jpg")]) 

    for img in image_list:
        image = cv2.imread(path + img)

        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image)

        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(path + img[:-4] + "_annotated.jpg", out.get_image()[:, :, ::-1])

finally:
    print("finished")
