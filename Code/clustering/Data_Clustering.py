from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image, ImageDraw
import os
from os.path import isfile, isdir

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


print("[INFO]\tProgramm started! Imports completed")

"""
FUNCTION - This function gets all ".jpg" and ".JPG" files in a directory and all of its subdirectories.
            A path to the directory and a set or list is required, to store the data - it's returning a sorted list
"""

def get_images(path, all_image_names):
    file_name_list = os.listdir(path)
    
    for element in file_name_list:
        if isfile(path + element) & (element.endswith(".jpg") or element.endswith(".JPG")):
            all_image_names.add((path + element).replace(base_path, ''))
        elif isdir(path + element):
            get_images(path + element + "/", all_image_names)

    all_image_names = list(all_image_names)
    all_image_names.sort()
    return all_image_names

"""
FUNCTION - This function places images in contentboxes and draws them on the position of their corresponding data point.
            The position and name of the actual Image, it's X and Y coordinates and the label are required - nothin will be returned. 
"""

def image_in_plot(i, name, X, Y, label):
    label_colors = {
        0 : "red",
        1 : "green",
        2 : "blue",
        3 : "purple",
        4 : "yellow",
        5 : "cyan",
        6 : "black",
        7 : "pink",
        8 : "gray",
        9 : "orange"
        }
    image = Image.open(base_path + name)
    frame = ImageDraw.Draw(image)
    frame.rectangle([(image.width, image.height), (5, 0)], outline=label_colors[label], width=50)
    image = image.resize((50, 75))
    ax = plt.gca()
    im = OffsetImage(image)
    im.image.axes = ax
    ab = AnnotationBbox(im, (X, Y), frameon=False, pad=0.0,)
    ax.add_artist(ab)


model = ResNet50(weights='imagenet', include_top=False)
# model.summary()

base_path = '/home/julius/PowerFolders/Masterarbeit/1_Datensaetze/personData200/'
clusters = 2

all_image_names = set()
all_image_names = get_images(base_path, all_image_names)

print("[INFO] {} Images were collected!".format(len(all_image_names)))

resnet_feature_dict = {}

for i, image_file in enumerate(all_image_names):
    img = image.load_img(base_path + image_file, target_size=(224, 224))

    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    resnet_feature = model.predict(img_data)
    np_resnet_feature = np.array(resnet_feature)
    resnet_feature_dict[image_file] = np_resnet_feature.flatten()

    print("[INFO]\tExtracted the Features of Image '{0: <25} ({1:5}/{2} | {3:6.2f}%)".format(image_file.split('\\')[-1], i+1, len(all_image_names), (i+1)/len(all_image_names)*100))

print("[INFO]\tFeatures Extracted! {0} Features in the format {1} were collected.".format(len(resnet_feature_dict), resnet_feature_dict[all_image_names[0]].shape))

features = np.array(list(resnet_feature_dict.values()))

print(features.shape)

pca = PCA(n_components=2, random_state=22)
pca.fit(features)

pca_features = pca.transform(features)

print(pca_features.shape)

kmeans = KMeans(n_clusters=clusters, random_state=22)
label_list = kmeans.fit_predict(pca_features)

print("[INFO]\tFinished clustering!")

labels_and_names = {}
for i, label in enumerate(label_list):
    if label not in labels_and_names:
        labels_and_names[label] = [all_image_names[i]]
    else:
        labels_and_names[label].append(all_image_names[i])

fig = plt.figure(figsize=(75, 75))

for label in range(clusters):
    filtered_label = pca_features[label_list == label]
    
    plt.scatter(filtered_label[:, 0], filtered_label[:, 1])

    for j, name in enumerate(labels_and_names[label]):
        image_in_plot(j, name, filtered_label[:, 0][j], filtered_label[:, 1][j], label)

    

# plt.legend()
plt.show()
