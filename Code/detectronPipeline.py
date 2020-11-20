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

image = cv2.imread("/home/julius/MA Arbeit/Images/jazzplakat.jpg", 1)

cv2.imshow("image", image)
cv2.waitKey(3000)



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
cv2.imshow("image", out.get_image()[:, :, ::-1])
k = cv2.waitKey(0) & 0xFF
if k == 32:
    cv2.destroyAllWindows()