import os
import json
import numpy as np

""" CLASS

Represents the COCO image data

Attributes: id (int), name (string), height (int), width (int)

Functions: print all attributes; return a dictionary with all attributes

"""


class Image:

    def __init__(self, id, name, height, width):
        self.id = id
        self.name = name
        self.height = height
        self.width = width

    def print(self):
        return '"height": {}\n,"width": {}\n,"id": {}\n,"file_name": {}\n'.format(
            self.height,
            self.width,
            self.id,
            self.name)

    def convertToDictionary(self):
        imagedict = {}
        imagedict["height"] = self.height
        imagedict["width"] = self.width
        imagedict["id"] = self.id
        imagedict["file_name"] = self.name

        return imagedict

""" CLASS

Represents the COCO category data

Attributes: id (int), supercategory (string), category (string)

Functions: print all attributes; return a dictionary with all attributes

"""

class Category:

    __doc__ = "The class, representing the categorie Data for the COCO Dataset"

    def __init__(self, id, supercategory, name):
        self.id = id
        self.name = name
        self.supercategory = supercategory

    def print(self):
        return '"Supercategory: {}\n"id": {}\n"category": {}\n'.format(
            self.supercategory,
            self.id,
            self.name)
    
    def convertToDictionary(self):
        categorydict = {}
        categorydict["supercategory"] = self.supercategory
        categorydict["id"] = self.id
        categorydict["name"] = self.name

        return categorydict


""" CLASS

Represents the COCO polygon data

Attributes: id (int), category_id (int), image_id (int), iscrowd (bool), segmentation (list), bbox (list), area (float)

Functions: print all attributes; return a dictionary with all attributes

"""


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
        return '"segmentation": {}\n"iscrowd": {}\n"area": {}\n"image_id": {}\n"bbox": {}\n"category_id": {}\n"id": {}\n'.format(
            self.segmentation,
            self.iscrowd,
            self.area,
            self.image_id,
            self.bbox,
            self.category_id,
            self.id)

    def convertToDictionary(self):
        polygondict = {}
        polygondict["segmentation"] = self.segmentation
        polygondict["iscrowd"] = self.iscrowd
        polygondict["area"] = self.area
        polygondict["image_id"] = self.image_id
        polygondict["bbox"] = self.bbox
        polygondict["category_id"] = self.category_id
        polygondict["id"] = self.id

        return polygondict

""" FUNCTION

Purpose: calculate the area of the polygon

Takes: A list with all x-coordinates and a list with all y-coordinates

Returns: The area of the given polygon

"""

def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

""" FUNCTION

Purpose: Extraction of the labelme data and restructuring for detectron2 COCO

Takes: Path to the '.json' and '.jpg' files

Returns: Nothing

"""

def main(DATASET_PATH):
    # helper variables
    json_dump = {}
    images = []
    categories = []
    polygons = []
    label_list = {}
    polygon_list = []

    # get all '.json' files in directory
    json_list = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".json")])

    # looping through the '.json' files
    # 
    # the first loop saves the general image data
    # the first enclosed loop saves the label data
    # the second enclosed loop saves the coordinates of the poligons
    for id_count, json_file in enumerate(json_list):
        with open(DATASET_PATH + json_file, "r") as content:
            data = json.load(content)
            
            image = Image(id_count, data["imagePath"].replace("../", ""), data["imageHeight"], data["imageWidth"])
            image_as_dict = image.convertToDictionary()
            images.append(image_as_dict)

            for shape_count, element in enumerate(data["shapes"]):
                if element["label"] not in label_list:
                    category = Category((len(label_list)), None, element["label"])
                    category_as_dict = category.convertToDictionary()
                    categories.append(category_as_dict)
                    label_list[element["label"]] = (len(label_list))

                x_coordinates = []
                y_coordinates = []

                # extract the polgon points
                for polygon in element["points"]:
                    x_coordinates.append(polygon[0])
                    y_coordinates.append(polygon[1])

                # transform into COCO format
                segmentation = []
                segmentation.append(list(sum(zip(x_coordinates, y_coordinates), ())))

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
                polygon_as_dict = polygon.convertToDictionary()
                polygons.append(polygon_as_dict)
                polygon_list.append(shape_count)

    # fill the dictionary to dump the data
    json_dump["images"] = images
    json_dump["categories"] = categories
    json_dump["annotations"] = polygons

    output_path = DATASET_PATH + "COCO_json/"
    if not os.DATASET_PATH.isdir(output_path):
        os.mkdir(output_path)

    with open(output_path + "output.json", "w") as output_file:
        json.dump(json_dump, output_file, indent=4)


if __name__ == "__main__":
    DATASET_PATH = "/home/julius/PowerFolders/Masterarbeit/1_Datensaetze/first_annotation_dataset/"
    main(DATASET_PATH)
