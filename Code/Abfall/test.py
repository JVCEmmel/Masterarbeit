import os
from os.path import isdir, isfile

def get_images(path, all_image_names):
    file_name_list = os.listdir(path)

    for element in file_name_list:
        if isfile(path+element) & (element.lower().endswith(".jpg")):
            all_image_names.add((path+element).replace(base_path, ''))
        elif isdir(path+element):
            get_images(path + element + "/", all_image_names)

    all_image_names = list(all_image_names)
    all_image_names.sort()
    return all_image_names

base_path = "/home/julius/Repos/Masterarbeit/LaTeX/Abbildungen/detections/05,04,2021-15,39,10/"

all_image_names = set()
all_image_names = get_images(base_path, all_image_names)

for element in all_image_names:
    new_name = element.replace(" ", "_")
    os.rename(base_path + element, base_path + new_name)
