from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset


import matplotlib.pyplot as plt
import random, cv2, time, os, shutil, ownlabelme2COCO, torch

print(torch.__version__)

""" FUNCTION

Purpose: Generate a train and a test dataset. The default is 5/1

Takes: Path to the images, the test and the train dataset directories. Split defines the ratio between test size and train size

Returns: Nothing

"""
def split_data (DATASET_PATH, train_set_path, test_set_path, split=5):
    os.mkdir(train_set_path)
    os.mkdir(test_set_path)

    # gather '.json' and '.jpg' files
    images = sorted([element for element in os.listdir(DATASET_PATH) if element.lower().endswith(".jpg")])
    jsons = sorted([element for element in os.listdir(DATASET_PATH) if element.endswith(".json")])

    # check, if they are even
    if len(images) != len(jsons):
        break

    # split
    for count in range(len(images)):
        if count % split == 0:
            shutil.move(DATASET_PATH + images[count], test_set_path + images[count])
            shutil.move(DATASET_PATH + jsons[count], test_set_path + jsons[count])
        else:
            shutil.move(DATASET_PATH + images[count], train_set_path + images[count])
            shutil.move(DATASET_PATH + jsons[count], train_set_path + jsons[count])
    
    # generate the COCO.json for detectron2
    ownlabelme2COCO.main(test_set_path)
    ownlabelme2COCO.main(train_set_path)


""" MAIN FUNCTION

Purpose: Train and test a detectron2 model

Takes: A detectron2 config

Returns: Nothing

"""

def train_and_evaluate(config):
    
    # train the model
    torch.cuda.empty_cache()
    trainer = DefaultTrainer(config)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # evaluate the model

    torch.cuda.empty_cache()
    evaluator = COCOEvaluator("test_set", config, distributed=False, output_dir=evaluation_path, use_fast_impl=False)
    test_loader = build_detection_test_loader(config, "test_set")
    inference_on_dataset(trainer.model, test_loader, evaluator)


    # export the config
    config_dump = config.dump()
    with open(config.OUTPUT_DIR + "config.yaml", "w+") as output_file:
        output_file.write(config_dump)


if __name__ == "__main__":

    ###SET WORK ENVIROMENT###
    WORK_DIR = "/home/julius/PowerFolders/Masterarbeit/"
    os.chdir(WORK_DIR)

    ###PATH TO DATASET###
    DATASET_PATH = "./1_Datensaetze/personData200/"

    # generate paths for testing and training
    train_set_path = DATASET_PATH + "train_split/"
    test_set_path = DATASET_PATH + "test_split/"

    # split data if it's not the case yet
    if not os.path.isdir(train_set_path):
        split_data(DATASET_PATH, train_set_path, test_set_path)

    # generate paths for outputs and make dirs
    starttime = time.strftime("%d,%m,%Y-%H,%M")
    model_path = "./trained_models/detectron2/{}/{}/".format(DATASET_PATH.split("/")[-2], starttime)
    evaluation_path = "./model_evaluation/detectron2/{}/{}/".format(DATASET_PATH.split("/")[-2], starttime)

    if not os.path.isdir(evaluation_path):
        os.makedirs(evaluation_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Import test and train dataset
    load_coco_json(train_set_path + "COCO_json/output.json", train_set_path, "train_set")
    register_coco_instances("train_set", {}, train_set_path + "COCO_json/output.json", train_set_path)
    train_set_metadata = MetadataCatalog.get("train_set")
    train_set_data = DatasetCatalog.get("train_set")

    load_coco_json(test_set_path + "COCO_json/output.json", test_set_path, "test_set")
    register_coco_instances("test_set", {}, test_set_path + "COCO_json/output.json", test_set_path)
    test_set_metadata = MetadataCatalog.get("test_set")
    test_set_data = DatasetCatalog.get("test_set")

    ###CONFIGURE THE MODEL###

    # load general config and merge with model config
    config = get_cfg()
    config.merge_from_file("/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # change Datasets for testing and training
    config.DATASETS.TRAIN = ("train_set", )
    config.DATASETS.TEST = ("test_set", )
    
    #config.DATALOADER.NUM_WORKERS = 2
    
    # load COCO trained weights 
    config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    
    # set output dir
    config.OUTPUT_DIR = model_path

    # set batchsize
    config.SOLVER.IMS_PER_BATCH = 2
    
    # set number of GPUs
    config.SOLVER.REFERENCE_WORLD_SIZE = 1

    # set learn rate
    config.SOLVER.BASE_LR = 0.02

    # set number of epochs
    config.SOLVER.MAX_ITER = 300

    # set RoIs per Image
    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # set number of classes
    config.MODEL.RETINANET.NUM_CLASSES = 11
    config.MODEL.ROI_HEADS.NUM_CLASSES = 11

    #####START#####
    train_and_evaluate(config)