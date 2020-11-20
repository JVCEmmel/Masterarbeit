import os
import json

class image():
    self.label= ""
    self.


path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"

#get all json files in directory
json_list = [f for f in os.listdir(path) if f.endswith(".json")]

print(json_list)

for id_count, json_file in enumerate(json_list):
    with open(path + json_file, "r") as content:
        data = json.load(content)
        for item in data["shapes"]:
            print(item["label"])