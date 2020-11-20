import json

with open("/home/julius/PowerFolders/Masterarbeit/Bilder/1 Datensätze/first_annotation_dataset/acht1_036.json") as json_file:
    data= json.load(json_file)

    print(data["imageHeight"])