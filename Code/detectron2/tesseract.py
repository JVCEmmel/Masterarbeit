from PIL import Image

import pytesseract, json, os, cv2
import numpy as np

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

###SET BASIC VIRABLES###
os.chdir("/home/julius/PowerFolders/Masterarbeit/")

dataset_path = "./1_Datensaetze/first_annotation_dataset/"
json_path = "./detections/first_annotation_dataset/14,04,2021-11,18/"
output_path = json_path


# gather images
images = sorted([element for element in os.listdir(dataset_path) if element.lower().endswith(".jpg")])

# read bounding boxes
with open(json_path + "bounding_boxes.json", "r+") as inputfile:
    bounding_boxes = json.load(inputfile)

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


# go over every image
for element in range(len(images)):
    # load image
    image = cv2.imread(dataset_path + images[element])
    dump_dict = {}

    # go over every box in the image
    for box in range(len(boxes_per_image[images[element]])):
        #get box cords, length and with and cut text out
        x_one = boxes_per_image[images[element]][box][0]
        x_two = boxes_per_image[images[element]][box][1]
        y_one = boxes_per_image[images[element]][box][2]
        y_two = boxes_per_image[images[element]][box][3]

        box_width = abs(x_one - y_one)
        box_height = abs(x_two - y_two)

        image_dict = {}
        image_dict["colored"] = image[x_two : x_two + box_height, x_one : x_one + box_width]
        image_dict["blue"], image_dict["green"], image_dict["red"] = pic_split(image_dict["colored"])
        image_dict["gray"] = cv2.cvtColor(image_dict["colored"], cv2.COLOR_BGR2GRAY)

        for variant in image_dict:
            text_dump = set()
            image_text = "{}".format(pytesseract.image_to_string(image_dict[variant], lang="deu"))
            image_text = image_text.replace("\x0c", "").split("\n")

            if len(image_text) > 0:
                [text_dump.add(text) for text in image_text if len(text) > 0]
            
            if variant not in dump_dict:
                dump_dict[variant] = list(text_dump)
            else:
                dump_dict[variant].append(list(text_dump))

    with open(output_path + "{}.json".format(images[element][:-4]), "w+", encoding="utf8") as output_file:
        json.dump(dump_dict, output_file, indent=4, ensure_ascii=False)

"""
    text_dump = pytesseract.image_to_string(image, lang="deu")

    with open(output_path + "{}_complete.txt".format(images[element][:-4]), "w+") as output_file:
        output_file.write(text_dump)
"""