from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog

import numpy as np
from os.path import isfile, isdir
from Prediction_analysis import prediction_analysis
import os, json, cv2, time, torch

###FUNCTION########################################################
# Yields every jpg-File in given directory and all subdirectories #
###################################################################

def get_images(base_path, all_image_names):
    file_name_list = os.listdir(base_path)
    
    for element in file_name_list:
        if isfile(base_path + element) & (element.lower().endswith(".jpg")):
            all_image_names.add((base_path + element).replace(path, ''))
        elif isdir(base_path + element):
            get_images(base_path + element + "/", all_image_names)

    all_image_names = list(all_image_names)
    all_image_names.sort()
    return all_image_names

###MAIN FUNCTION#############
# Predicts a list of images #
############################# 

def serial_Predictor(config, path, save_images = False):
    # Generate path and variables for output 
    output_dict = {}
    all_output_dict = {}
    output_path = "/".join(path.split("/")[:-3]) + "/detections/{}/{}/".format(path.split("/")[-2], time.strftime("%d,%m,%Y-%H,%M"))
    if not isdir(output_path):
        os.makedirs(output_path)

    # Get all images
    image_list = set()
    image_list = get_images(path, image_list)

    ###CONSOLE OUTPUT###
    print("[INFO] {} Images were collected".format(len(image_list)))

    # go through the list of images
    for count, element in enumerate(image_list):

        # load image and get prediction
        torch.cuda.empty_cache()
        picture = cv2.imread(path + element)
        predictor = DefaultPredictor(config)
        outputs = predictor(picture)

        # extract esssential informations
        predicted_classes = list(np.asarray(outputs["instances"].pred_classes.to("cpu")))
        prediction_scores =  list(np.asarray(outputs["instances"].scores.to("cpu")))
        prediction_scores = [float(element) for element in prediction_scores]
        thing_classes = MetadataCatalog.get(config.DATASETS.TRAIN[0]).thing_classes
        predicted_classes_names = [thing_classes[element] for element in predicted_classes]

        output_dict[element] = {"category_names": predicted_classes_names, "prediction_scores": prediction_scores}
        all_output_dict[element] = outputs

        ###CONSOLE OUTPUT###
        print("[INFO] Predicted {}\t{}/{}".format(element, count+1, len(image_list)))

        # Save images if it's wanted
        if save_images:
            v = Visualizer(picture[:,:, ::-1], MetadataCatalog.get(config.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            cv2.imwrite(output_path + element[:-4] + "_with_detections.jpg", out.get_image()[:, :, ::-1])
    
    # save essential prediciton informations
    with open(output_path + "detections.json", "w+") as output_file:
        json.dump(output_dict, output_file)

    # save all outputs
    with open(output_path + "detections_complete.txt", "w+") as output_file:
        output_file.write(str(all_output_dict))
    
    # analyse predictions
    prediction_analysis(output_path)

if __name__ == "__main__":

    ###SET WORK ENVIROMENT###
    work_dir = "/home/julius/PowerFolders/Masterarbeit/"
    os.chdir(work_dir)

    ###PATH TO DATASET###
    path = "./1_Datensaetze/data100/"
    dataset_path = "./1_Datensaetze/personData200/"
    train_set_path = dataset_path + "train_split/"

    ###CONFIG FOR MODEL###
    config = get_cfg()
    config.merge_from_file("./trained_models/detectron2/personData200/06,04,2021-21,27/config.yaml")
    # config.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    config.MODEL.WEIGHTS = "./trained_models/detectron2/personData200/06,04,2021-21,27/model_final.pth"
    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 20

    load_coco_json(train_set_path + "COCO_json/output.json", train_set_path, "train_set")
    register_coco_instances("train_set", {}, train_set_path + "COCO_json/output.json", train_set_path)
    train_set_metadata = MetadataCatalog.get("train_set")

    ###START###
    serial_Predictor(config, path, save_images=True)