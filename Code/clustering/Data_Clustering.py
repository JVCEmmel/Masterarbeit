from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from keras.preprocessing import image as keras_img

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image, ImageDraw
import os
import time
from os.DATASET_PATH import isfile, isdir

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

""" FUNCTION

Purpose: Extracting all '.jpg' files

Takes: The DATASET_PATH to the directory in which the files are located; a set with all collected file names

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

Purpose: places images according to their coordinates in a grid

Takes: Image name, X and Y coordinates and the current label

Returns: Nothing

"""

def image_in_plot(name, X, Y, label):
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
        9 : "orange",
        10: "brown",
        11: "lime",
        12: "dodgerblue",
        13: "turquoise",
        14: "magenta",
        15: "navy",
        16: "khaki",
        17: "lightgray"
        }
    image = Image.open(DATASET_PATH + name)
    frame = ImageDraw.Draw(image)
    frame.rectangle([(image.width, image.height), (5, 0)], outline=label_colors[label], width=50)
    image = image.resize((50, 75))
    ax = plt.gca()
    im = OffsetImage(image)
    im.image.axes = ax
    ab = AnnotationBbox(im, (X, Y), frameon=False, pad=0.0,)
    ax.add_artist(ab)


###SET BASIC VARIABLES###
WORK_DIR = "/home/julius/PowerFolders/Masterarbeit/"
os.chdir(WORK_DIR)

DATASET_PATH = './1_Datensaetze/personData200/'
OUTPUT_PATH = './cluster_outputs/{}/'.format(time.strftime("%d,%m,%Y-%H,%M,%S"))
os.mkdir(OUTPUT_PATH)
CLUSTERS = 17

# load model
model = ResNet50(weights='imagenet', include_top=False)

# load trained model
"""
model = ResNet50(include_top=False)
model.load_weights("./trained_models/tensorflow/Checkpoint-29,03,2021-21,50,10")
"""

# gather all images
all_image_names = set()
all_image_names = get_images(DATASET_PATH, all_image_names)

print("[INFO] {} Images were collected!".format(len(all_image_names)))

# extract the features of all images
resnet_feature_dict = {}
for i, image_file in enumerate(all_image_names):
    img = keras_img.load_img(DATASET_PATH + image_file, target_size=(224, 224))

    img_data = keras_img.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    resnet_feature = model.predict(img_data)
    np_resnet_feature = np.array(resnet_feature)
    resnet_feature_dict[image_file] = np_resnet_feature.flatten()

    print("[INFO]\tExtracted the Features of Image '{0: <25} ({1:5}/{2} | {3:6.2f}%)".format(image_file.split('\\')[-1], i+1, len(all_image_names), (i+1)/len(all_image_names)*100))

print("[INFO]\tFeatures Extracted! {0} Features in the format {1} were collected.".format(len(resnet_feature_dict), resnet_feature_dict[all_image_names[0]].shape))

# transform features
features = np.array(list(resnet_feature_dict.values()))

print("[INFO]\tFeatures were brought to shape {}".format(features.shape))

pca = PCA(n_components=2, random_state=22)
pca.fit(features)
pca_features = pca.transform(features)

print("[INFO]\tFeatures were transformed into the shape of {}".format(pca_features.shape))

# cluster images
kmeans = KMeans(n_clusters=CLUSTERS, random_state=22)
label_list = kmeans.fit_predict(pca_features)

print("[INFO]\tFinished clustering!")

# resort data
labels_and_names = {}
for i, label in enumerate(label_list):
    if label not in labels_and_names:
        labels_and_names[label] = [all_image_names[i]]
    else:
        labels_and_names[label].append(all_image_names[i])

# save only the scatter plot

scatter_fig = plt.figure(figsize=(20,20))

for label in range(CLUSTERS):
    filtered_label = pca_features[label_list == label]
    
    plt.scatter(filtered_label[:, 0], filtered_label[:, 1])
    for i, name in enumerate(labels_and_names[label]):
        plt.text(filtered_label[:, 0][i], filtered_label[:, 1][i], name)
    
plt.savefig(OUTPUT_PATH + "_scatter.png")

# save the plot with all images combined

image_fig = plt.figure(figsize=(20, 20))

for label in range(CLUSTERS):
    filtered_label = pca_features[label_list == label]
    plt.scatter(filtered_label[:, 0], filtered_label[:, 1])
    plt.axis("off")
    for j, name in enumerate(labels_and_names[label]):
        image_in_plot(name, filtered_label[:, 0][j], filtered_label[:, 1][j], label)

plt.savefig(OUTPUT_PATH + "_image.png")

# Save all cluster figures seprately

sub_fig = plt.figure(figsize=(20, 20))

for label in range(CLUSTERS):
    ax = plt.subplot(5, 5, label+1)
    filtered_label = pca_features[label_list == label]
    plt.scatter(filtered_label[:, 0], filtered_label[:, 1])
    plt.axis("off")
    for j, name in enumerate(labels_and_names[label]):
        image_in_plot(name, filtered_label[:, 0][j], filtered_label[:, 1][j], label)
    
    cut_out = ax.get_window_extent().transformed(sub_fig.dpi_scale_trans.inverted())
    plt.savefig(OUTPUT_PATH + "label_{}_cluster.png".format(label), bbox_inches=cut_out)

# save labels and names
with open(output_path + "labels_and_names.json", "w") as output_file:
    json.dump(labels_and_names, output_file, indent=4)