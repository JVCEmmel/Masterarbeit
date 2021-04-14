import pytesseract, json, os, cv2
import numpy as np


os.chdir("/home/julius/PowerFolders/Masterarbeit/")

dataset_path = "./1_Datensaetze/first_annotation_dataset/"
json_path = "./detections/first_annotation_dataset/14,04,2021-11,18/"
output_path = json_path

images = sorted([element for element in os.listdir(dataset_path) if element.lower().endswith(".jpg")])

with open(json_path + "bounding_boxes.json", "r+") as inputfile:
    bounding_boxes = json.load(inputfile)

boxes_per_image = {}
for count in range(len(images)):
    current_image = bounding_boxes[images[count]]

    for count_two in range(len(current_image["category_names"])):
        if current_image["category_names"][count_two] == "text":
            if images[count] not in boxes_per_image:
                boxes_per_image[images[count]] = [current_image["prediction_boxes"][count_two]]
            else:
                boxes_per_image[images[count]].append(current_image["prediction_boxes"][count_two])

print(boxes_per_image[images[0]][0])


for element in range(len(images)):
    image = cv2.imread(dataset_path + images[element])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #first_pixel = image[0, 0]
    #image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    text_dump = set()
    for box in range(len(boxes_per_image[images[element]])):
        x_one = boxes_per_image[images[element]][box][0]
        x_two = boxes_per_image[images[element]][box][1]
        y_one = boxes_per_image[images[element]][box][2]
        y_two = boxes_per_image[images[element]][box][3]

        box_width = abs(x_one - y_one)
        box_height = abs(x_two - y_two)

        box_image = image[x_two : x_two + box_height, x_one : x_one + box_width]

        image_text = "{}".format(pytesseract.image_to_string(box_image, lang="deu"))
        image_text = image_text.split("\n")

        if len(image_text) > 0:
            [text_dump.add(element) for element in image_text]
        """
        cv2.imshow("Bild", box_image)
        cv2.waitKey(20000)
        cv2.destroyAllWindows()
        """
        """
        cv2.imwrite(output_path + "image_{}_box_{}.jpg".format(element, box), box_image)
        """
    with open(output_path + "{}.txt".format(images[element][:-4]), "w+") as output_file:
        output_file.write(str(text_dump))

    text_dump = pytesseract.image_to_string(image, lang="deu")

    with open(output_path + "{}_complete.txt".format(images[element][:-4]), "w+") as output_file:
        output_file.write(text_dump)
    """
    cv2.imwrite(output_path + "test{}.jpg".format(element), image)

    display_image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
    cv2.imshow("Bild", display_image)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
    """