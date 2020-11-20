import os
import json

class Image:

    __doc__="The class, representing the image Data for the COCO Dataset"

    def __init__(self, id, name, height, width):
        self.id = id
        self.name = name
        self.height = height
        self.width = width

    def print(self):
        print("Image: {}\nID: {}\nWidth: {}\nHeight: {}\n".format(self.name, self.id, self.width, self.height))


class Category:

    __doc__="The class, representing the categorie Data for the COCO Dataset"

    def __init__(self, id, supercategory, name):
        self.id = id
        self.name = name
        self.supercategory = supercategory

    def print(self):
        print("Category: {}\nID: {}\nSupercategory: {}\n".format(self.name, self.id, self.supercategory))


# helper variables -
# 
# path gives the directory with images and .json,
# images and categories store the classes created,
# label_list keeps track over the gathered labels

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"
images = []
categories = []
label_list = []

# get all json files in directory
json_list = [f for f in os.listdir(path) if f.endswith(".json")]

# looping through the .json files
# 
# the first loop saves the general image data
# the first enclosed loop saves the label data
# the second enclosed loop saves the coordinates of the poligons
for id_count, json_file in enumerate(json_list):
    with open(path + json_file, "r") as content:
        data = json.load(content)
        
        image = Image(id_count, json_file, data["imageHeight"], data["imageWidth"])
        images.append(image)

        for element in data["shapes"]:
            if element["label"] not in label_list:
                category = Category((len(label_list)), None, element["label"])
                categories.append(category)
                label_list.append(element["label"])


for element in categories:
    element.print()
