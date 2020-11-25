from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2

try:
    path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"

    load_coco_json(path + "output.json", path[:-1], "first_anno_ds")

    register_coco_instances("first_anno_ds", {}, path + "output.json", path[:-1])

    first_anno_ds_metadata = MetadataCatalog.get("first_anno_ds")

    print(first_anno_ds_metadata)

    dataset_dicts = DatasetCatalog.get("first_anno_ds")

    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:,:,::-1], metadata=first_anno_ds_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        
        cv2.imshow("Bild", vis.get_image()[:,:,::-1])
        cv2.waitKey(3000)
        cv2.destroyWindow("Bild")

finally:
    cv2.destroyAllWindows()