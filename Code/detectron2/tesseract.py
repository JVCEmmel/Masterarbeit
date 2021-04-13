import pytesseract, json, os, cv2

os.chdir("/home/julius/PowerFolders/Masterarbeit/")

dataset_path = "./1_Datensaetze/first_annotation_dataset/"
json_path = "./detections/first_annotation_dataset/09,04,2021-15,29 (personData200 - Threshold 0.7)/"
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
    image = cv2.resize(image, (2227, 3340))

    for box in range(len(boxes_per_image[images[element]])):
        x_one = int(boxes_per_image[images[element]][box][0])
        y_one = int(boxes_per_image[images[element]][box][1])
        x_two = int(boxes_per_image[images[element]][box][2])
        y_two = int(boxes_per_image[images[element]][box][3])

        #cv2.rectangle(image, (x_one, x_two), (y_one, y_two), (0, 0, 255))
        cv2.rectangle(image, (x_two, x_one), (y_two, y_one), (0, 255, 0))

    display_image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
    """
    cv2.imshow("Bild", display_image)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
    """

    cv2.imwrite(output_path + "test{}.jpg".format(element), image)