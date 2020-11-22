import os
import json
import numpy as np

class Image:

    __doc__ = "The class, representing the image Data for the COCO Dataset"

    def __init__(self, id, name, height, width):
        self.id = id
        self.name = name
        self.height = height
        self.width = width

    def print(self):
        return "Image: {}\nID: {}\nWidth: {}\nHeight: {}\n".format(
            self.name,
            self.id,
            self.width,
            self.height)


class Category:

    __doc__ = "The class, representing the categorie Data for the COCO Dataset"

    def __init__(self, id, supercategory, name):
        self.id = id
        self.name = name
        self.supercategory = supercategory

    def print(self):
        return "Category: {}\nID: {}\nSupercategory: {}\n".format(
            self.name,
            self.id,
            self.supercategory)


class Polygon:
    
    __doc__ = "The class, representing the categorie Data for the COCO Dataset"

    def __init__(self, id, category_id, image_id, iscrowd, segmentation, bbox, area):
        self.id = id
        self.category_id = category_id
        self.image_id = image_id
        self.iscrowd = iscrowd
        self.segmentation = segmentation
        self.bbox = bbox
        self.area = area

    def print(self):
        return "Polygon ID: {}\nImage ID: {}\nCategory ID: {}\niscrowd: {}\nAreasize: {}\nCooridnates: {}\nBoundingbox: {}\n".format(
            self.id,
            self.image_id,
            self.category_id,
            self.iscrowd,
            self.area,
            self.segmentation,
            self.bbox)


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

# helper variables -
# 
# path gives the directory with images and .json,
# images and categories store the classes created,
# label_list keeps track over the gathered labels

path = "/home/julius/PowerFolders/Masterarbeit/Bilder/1_Datensaetze/first_annotation_dataset/"
images = []
categories = []
polygons = []
label_list = {}
polygon_list = []

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

        for shape_count, element in enumerate(data["shapes"]):
            if element["label"] not in label_list:
                category = Category((len(label_list)), None, element["label"])
                categories.append(category)
                label_list[element["label"]] = (len(label_list))

            x_coordinates = []
            y_coordinates = []
            for polygon in element["points"]:
                x_coordinates.append(polygon[0])
                y_coordinates.append(polygon[1])

            # segmentation = list(zip(x_coordinates, y_coordinates))
            segmentation = list(sum(zip(x_coordinates, y_coordinates), ()))

            # get the values of the bbox
            smallest_x = int(min(x_coordinates))
            smallest_y = int(min(y_coordinates))
            biggest_x = int(max(x_coordinates))
            biggest_y = int(max(y_coordinates))

            bbox_height = biggest_y-smallest_y
            bbox_width = biggest_x-smallest_x

            bbox = [smallest_x, smallest_y, bbox_width, bbox_height]

            # get the area of the polygon
            polygon_area = PolyArea(x_coordinates, y_coordinates)

            # create polygon instance and add it to the list
            polygon = Polygon(len(polygon_list), label_list[element["label"]], id_count, 0, segmentation, bbox, polygon_area)
            polygons.append(polygon)
            polygon_list.append(shape_count)

with open(path + "output.txt", "a") as output_file:
    for element in images:
        print(element.print(), file=output_file)
    for element in categories:
        print(element.print(), file=output_file)
    for element in polygons:
        print(element.print(), file=output_file)