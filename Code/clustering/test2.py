import os
from os.path import isfile, isdir

base_path = 'E:\\PowerFolders\\Masterarbeit\\Bilder\\'
# path = 'E:\\PowerFolders\\Masterarbeit\\Bilder\\'
all_image_names = set()

def get_images(path, all_image_names):
    file_name_list = os.listdir(path)
    
    for element in file_name_list:
        if isfile(path + element) & (element.endswith(".jpg") or element.endswith(".JPG")):
            all_image_names.add((path + element).replace(base_path, ''))
        elif isdir(path + element):
            get_images(path + element + "\\", all_image_names)

    all_image_names = list(all_image_names)
    all_image_names.sort()
    return all_image_names

all_image_names = get_images(base_path, all_image_names)

print(all_image_names)
print(len(all_image_names))
print(len(max(all_image_names)))


"""
path = 'E:\\PowerFolders\\Masterarbeit\\1_Datensaetze\\first_annotation_dataset\\'
full_path = 'E:\\PowerFolders\\Masterarbeit\\1_Datensaetze\\first_annotation_dataset\\acht1_036.jpg'

print(full_path.split('\\')[-1])
"""