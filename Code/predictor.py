from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, cv2, helpfulFunctions

def serialPredictor(config, path, test_set_data, test_set_metadata):
    config.MODEL.WEIGHTS = os.path.join(config.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(config)

    for counter, data in enumerate(test_set_data):
        image = cv2.imread(data["file_name"])
        prediction = predictor(image)
        visualizer = Visualizer(image, metadata=test_set_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        visualzation = visualizer.draw_instance_predictions(prediction["instances"].to("cpu"))

        if os.path.isdir(path + "predictions") == False:
            os.makedirs(path + "predictions")
        cv2.imwrite(path + "predictions/image" + str(counter) + ".jpg", visualzation.get_image())


if __name__ == "__main__":
    path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/data100/"
    test_path = path + "test_split/"
    dataset_name = path.split("/")[-2]
    load_coco_json(test_path + "output.json", test_path[:-1], "test_set")
    register_coco_instances("test_set", {}, test_path + "output.json", test_path[:-1])
    test_set_metadata = MetadataCatalog.get("test_set")
    test_set_data = DatasetCatalog.get("test_set")

    config = get_cfg()
    config.merge_from_file("/home/julius/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # customize the config
    config.DATASETS.TRAIN = ("train_set",)
    config.DATASETS.TEST = ("test_set",)
    config.DATALOADER.NUM_WORKERS = 2
    config.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    config.OUTPUT_DIR = "/home/julius/PowerFolders/Masterarbeit/detectron_training_output/" + dataset_name + "/"
    config.SOLVER.IMS_PER_BATCH = 2
    config.SOLVER.BASE_LR = 0.02
    config.SOLVER.MAX_ITER = (300)
    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
    config.MODEL.ROI_HEADS.NUM_CLASSES = 16
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    serialPredictor(config, path, test_set_data, test_set_metadata)
