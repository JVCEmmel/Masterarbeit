import csv
import shutil
import os, sys

try:
    # basic variables
    path = "/home/julius/PowerFolders/Masterarbeit/"
    dataset_name = "test"
    csv_matrix = []
    picture_names = []

    # opening the cvs file and save it into matrix
    with open(path + "Listen von Datensätzen/Datensatzliste.CSV") as csv_file:
        csv_reader_objekt = csv.reader(csv_file, delimiter=';')

        for row in csv_reader_objekt:
            csv_matrix.append(row)

    # getting picture file names out of matrix 
    for element in csv_matrix:
        if element[1] == "":
            picture_names.append(element[2])
        elif element[2] == "":
            picture_names.append(element[1])

    # extracting the name of the image directory
    dir_name_list = [letter for letter in picture_names[0] if letter.isalpha()]
    dir_name = ""
    for element in dir_name_list:
        dir_name += element

    # complete the filenames and sort empty filenames
    picture_names = [element + ".JPG" for element in picture_names]
    picture_names = [element for element in picture_names if element != ".JPG"]

    # create destination directory
    if os.path.isdir(path + "Bilder/1_Datensaetze/" + dataset_name) == False:
        os.makedirs(path + "Bilder/1_Datensaetze/" + dataset_name)

    # copy pictures
    for element in picture_names:
        shutil.copyfile(path + "Bilder/" + dir_name + "/" + element, path + "Bilder/1_Datensaetze/" + dataset_name + "/" + element)

finally:
    print(str(len([name for name in os.listdir(path + "Bilder/1_Datensaetze/" + dataset_name)])) + " out of " + str(len(picture_names)) + " had been copied")
