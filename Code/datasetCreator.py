import csv
import shutil
import os, sys

def copyImages (path, dataset_name, dir_name, images_to_copy):
    if os.path.isdir(path + "Bilder/1_Datensaetze/" + dataset_name) == False:
        os.makedirs(path + "Bilder/1_Datensaetze/" + dataset_name)
    else:
        copied_images = sorted([element for element in os.listdir(path + "Bilder/1_Datensaetze/" + dataset_name) if (element.endswith(".JPG")) or element.endswith(".jpg")])
        images_to_copy = [image for image in images_to_copy if image not in copied_images]
    
    [shutil.copyfile(path + "Bilder/" + dir_name + "/" + element, path + "Bilder/1_Datensaetze/" + dataset_name + "/" + element) for element in images_to_copy]


def getCopyImages(path, dataset_name, image_names):
    dir_name_list = [letter for letter in image_names[0] if letter.isalpha()]
    dir_name = ""
    for element in dir_name_list:
        dir_name += element

    # get the files to copy them
    images = os.listdir(path + "Bilder/" + dir_name)

    images_to_copy = []
    for image in images:
        for name in image_names:
            if name in image:
                images_to_copy.append(image)
            if len(images_to_copy) == len(image_names):
                break
    images_to_copy = sorted(images_to_copy)

    copyImages(path, dataset_name, dir_name, images_to_copy)

def extractCSV(path, dataset_name):
    csv_matrix = []
    with open(path + "Listen von Datensätzen/" + dataset_name + ".CSV") as csv_file:
        csv_reader_objekt = csv.reader(csv_file, delimiter=';')

        for row in csv_reader_objekt:
            csv_matrix.append(row)
    
    image_names = []
    for element in csv_matrix[1:]:
        if element[1] != "":
            image_names.append(element[1])
        elif element[2] != "":
            image_names.append(element[2])
        elif (element[1] == "") & (element[2] == ""):
            pass
    
    getCopyImages(path, dataset_name, image_names)


try:
    if if __name__ == "__main__": 
        path = "/home/julius/PowerFolders/Masterarbeit/"
        dataset_name = "data100"
        extractCSV(path, dataset_name)
    
finally:
    print("[INFO] Process F" + str(len([name for name in os.listdir(path + "Bilder/1_Datensaetze/" + dataset_name) if name.endswith(".JPG") or name.endswith(".jpg)")])) + " out of " + str(len(image_names)) + " had been copied")
