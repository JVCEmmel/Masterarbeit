from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset


import matplotlib.pyplot as plt
import random, cv2, time, os, shutil, ownlabelme2COCO, predictor, torch

print(torch.__version__)

# set paths

dataset_name = "personData200"
path = "/home/julius/Schreibtisch/test_dir/"

dataset_path = path + "1_Datensaetze/{}/".format(dataset_name)
train_set_path = dataset_path + "train_split/"
test_set_path = dataset_path + "test_split/"

starttime = time.strftime("%d,%m,%Y-%H,%M")
model_path = path + "trained_models/detectron2/{}/{}/".format(dataset_name, starttime)
config_path = model_path + "config.yaml"
evaluation_path = path + "model_evaluation/detectron2/{}/{}/".format(dataset_name, starttime)


# split the data if its not the case
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


# Import test and train dataset
load_coco_json(train_set_path + "COCO_json/output.json", train_set_path, "train_set")
register_coco_instances("train_set", {}, train_set_path + "COCO_json/output.json", train_set_path)
train_set_metadata = MetadataCatalog.get("train_set")
train_set_data = DatasetCatalog.get("train_set")

load_coco_json(test_set_path + "COCO_json/output.json", test_set_path, "test_set")
register_coco_instances("test_set", {}, test_set_path + "COCO_json/output.json", test_set_path)
test_set_metadata = MetadataCatalog.get("test_set")
test_set_data = DatasetCatalog.get("test_set")


# configure the model parameters
config = get_cfg()
config.merge_from_file("/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
config.DATASETS.TRAIN = ("train_set",)
config.DATASETS.TEST = ("test_set",)
config.DATALOADER.NUM_WORKERS = 2
config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
config.OUTPUT_DIR = model_path
config.SOLVER.IMS_PER_BATCH = 2
config.SOLVER.REFERENCE_WORLD_SIZE = 1
config.SOLVER.BASE_LR = 0.02
config.SOLVER.MAX_ITER = 150
config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
config.MODEL.ROI_HEADS.NUM_CLASSES = 16
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# train the model
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()
trainer = DefaultTrainer(config)
trainer.resume_or_load(resume=False)
trainer.train()

# evaluate the model
if not os.path.isdir(evaluation_path):
        os.makedirs(evaluation_path)
evaluator = COCOEvaluator("test_set", config, distributed=False, output_dir=evaluation_path, use_fast_impl=False)
torch.cuda.empty_cache()
validation_loader = build_detection_test_loader(config, "test_set")
inference_on_dataset(trainer.model, validation_loader, evaluator)


# export the config
config.dump()