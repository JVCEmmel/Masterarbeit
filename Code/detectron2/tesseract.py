from PIL import Image
from os.path import isfile, isdir

import pytesseract, json, os, cv2
import numpy as np

""" FUNCTION

Purpose: Extracting all '.jpg' files

Takes: The path to the directory in which the files are located; a set with all collected file names

Returns: A sorted list of all '.jpg' file names

"""

def get_images(base_path, all_image_names):
    file_name_list = os.listdir(base_path)
    
    for element in file_name_list:
        if isfile(base_path + element) & (element.lower().endswith(".jpg")):
            all_image_names.add((base_path + element).replace(DATASET_PATH, ''))
        elif isdir(base_path + element):
            get_images(base_path + element + "/", all_image_names)

    all_image_names = list(all_image_names)
    all_image_names.sort()
    return all_image_names

""" FUNCTION

Purpose: Splitting an image in its three color channels

Takes: An image

Returns: Three images for each color channel

"""

def pic_split(img):

    #converting into numpy
    img_array = np.array(img).astype(float)

    #the actual split
    img_array_blue = img_array[:,:,0]
    img_array_green = img_array[:,:,1]
    img_array_red = img_array[:,:,2]

    np.where(img_array_blue<128, 1, 0)
    np.where(img_array_green<128, 1, 0)
    np.where(img_array_red<128,1 ,0)

    #converting them back
    img_blue = Image.fromarray(img_array_blue.astype(np.uint8))
    img_green = Image.fromarray(img_array_green.astype(np.uint8))
    img_red = Image.fromarray(img_array_red.astype(np.uint8))

    return img_blue, img_green, img_red

###CONSOLE OUTPUT###
print("[INFO] Programm started!")

###SET BASIC VIRABLES###
WORK_DIR = "/home/julius/PowerFolders/Masterarbeit/"
os.chdir(WORK_DIR)

DATASET_PATH = "./1_Datensaetze/data100/"
JSON_PATH = "./detections/data100/16,04,2021-12,43/"

###CONSOLE OUTPUT###
print("[INFO] Collecting images in '{}'.".format(DATASET_PATH))

# gather images
images = set()
images = get_images(DATASET_PATH, images)

###CONSOLE OUTPUT###
print("[INFO] Reading bounding box export.")

# read bounding boxes
with open(JSON_PATH + "bounding_boxes.json", "r+") as inputfile:
    bounding_boxes = json.load(inputfile)

###CONSOLE OUTPUT###
print("[INFO] Restructuring data.")

# restructure boxes per image
boxes_per_image = {}
for count in range(len(images)):
    current_image = bounding_boxes[images[count]]

    for count_two in range(len(current_image["category_names"])):
        if current_image["category_names"][count_two] == "text":
            if images[count] not in boxes_per_image:
                boxes_per_image[images[count]] = [current_image["prediction_boxes"][count_two]]
            else:
                boxes_per_image[images[count]].append(current_image["prediction_boxes"][count_two])

    # if theres no text box in image, create an empty list
    if images[count] not in boxes_per_image:
        boxes_per_image[images[count]] = []

###CONSOLE OUTPUT###
print("[INFO] Beginning OCR!")

# go over every image
for element in range(len(images)):
    # load image
    image_dict = {}
    image_dict["complete_image"] = cv2.imread(DATASET_PATH + images[element])
    dump_dict = {}

    ###CONSOLE OUTPUT###
    print("[INFO] Read Image: {}.".format(images[element]))

    # go over every box in the image
    for box in range(len(boxes_per_image[images[element]])):
        # get box cords, length and with and cut text out
        x_one = boxes_per_image[images[element]][box][0]
        x_two = boxes_per_image[images[element]][box][1]
        y_one = boxes_per_image[images[element]][box][2]
        y_two = boxes_per_image[images[element]][box][3]

        box_width = abs(x_one - y_one)
        box_height = abs(x_two - y_two)

        image_dict["colored_box"] = image_dict["complete_image"][x_two : x_two + box_height, x_one : x_one + box_width]
        image_dict["blue_box"], image_dict["green_box"], image_dict["red_box"] = pic_split(image_dict["colored_box"])
        image_dict["gray_box"] = cv2.cvtColor(image_dict["colored_box"], cv2.COLOR_BGR2GRAY)

        # OCR on every variant
        for variant in image_dict:
            text_dump = set()

            if (variant != "complete_image"):
                image_text = "{}".format(pytesseract.image_to_string(image_dict[variant], lang="deu"))
                image_text = image_text.replace("\x0c", "").split("\n")
            else:
                continue

            if len(image_text) > 0:
                [text_dump.add(text) for text in image_text if len(text) > 0]
            
            if variant not in dump_dict:
                dump_dict[variant] = list(text_dump)
            else:
                dump_dict[variant].append(list(text_dump))
            
        ###CONSOLE OUTPUT###
        print("[INFO] Finished Detection for box {}/{} of image {}".format(box+1, len(boxes_per_image[images[element]]), images[element]))

    # OCR on the whole image
    image_text = "{}".format(pytesseract.image_to_string(image_dict["complete_image"], lang="deu"))
    image_text = image_text.replace("\x0c", "").split("\n")
    if len(image_text) > 0:
        [text_dump.add(text) for text in image_text if len(text) > 0]
    dump_dict["complete_image"] = list(text_dump)

    # dump results
    with open(JSON_PATH + "{}.json".format(images[element].split("/")[-1][:-4]), "w+", encoding="utf8") as output_file:
        json.dump(dump_dict, output_file, indent=4, ensure_ascii=False)
    
    ###CONSOLE OUTPUT###
    print("[INFO] Exported OCR data for image: {}.".format(images[element]))