from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"

register_coco_instances("first_anno_ds", {}, path + "output.json", path[:-1])

first_anno_ds_metadata = MetadataCatalog.get("first_anno_ds")

print(first_anno_ds_metadata)