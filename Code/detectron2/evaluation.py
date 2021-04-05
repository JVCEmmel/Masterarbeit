from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import time, os, torch


dataset_name = "personData200"
path = "/home/julius/Schreibtisch/test_dir/"
dataset_path = path + "1_Datensaetze/{}/".format(dataset_name)
train_set_path = dataset_path + "train_split/"
test_set_path = dataset_path + "test_split/"

starttime = time.strftime("%d,%m,%Y-%H,%M")
model_path = path + "trained_models/detectron2/{}/05,04,2021-12,04/".format(dataset_name)
config_path = model_path + "config.yaml"
evaluation_path = path + "model_evaluation/detectron2/{}/{}/".format(dataset_name, starttime)

load_coco_json(train_set_path + "COCO_json/output.json", train_set_path, "train_set")
register_coco_instances("train_set", {}, train_set_path + "COCO_json/output.json", train_set_path)
train_set_metadata = MetadataCatalog.get("train_set")
train_set_data = DatasetCatalog.get("train_set")

load_coco_json(test_set_path + "COCO_json/output.json", test_set_path, "test_set")
register_coco_instances("test_set", {}, test_set_path + "COCO_json/output.json", test_set_path)
test_set_metadata = MetadataCatalog.get("test_set")
test_set_data = DatasetCatalog.get("test_set")

config = get_cfg()
config.merge_from_file(config_path)
config.DATASETS.TRAIN = ("train_set")
config.DATASETS.TEST = ("test_set",)
config.WEIGHTS = model_path + "model_final.pth"

# print(config)

if not os.path.isdir(evaluation_path):
        os.makedirs(evaluation_path)
evaluator = COCOEvaluator("test_set", config, distributed=False, output_dir=evaluation_path, use_fast_impl=False)
torch.cuda.empty_cache()
trainer = DefaultTrainer(config)
validation_loader = build_detection_test_loader(config, "test_set")
inference_on_dataset(trainer.model, validation_loader, evaluator)
#trainer.test(config, trainer.model, evaluators=evaluator)

#predictor.serialPredictor(config, path, test_set_data, test_set_metadata)
