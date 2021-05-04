import matplotlib.pyplot as plt
import numpy as np
import os, time
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import Sequential
from keras.models import Model

###SET BASIC VARIABLES###
WORK_DIR = "/home/julius/PowerFolders/Masterarbeit/"
os.chdir(WORK_DIR)

DATASET_PATH = "./1_Datensaetze/tensorflow"
OUTPUT_PATH = "./trained_models/tensorflow/"

# import datasets
train_dataset = image_dataset_from_directory(DATASET_PATH, validation_split=0.25, subset="training", seed=123, image_size=(160, 160), batch_size=25)
validation_dataset = image_dataset_from_directory(DATASET_PATH, validation_split=0.25, subset="validation", seed=123, image_size=(160, 160), batch_size=25)

class_names = train_dataset.class_names

validation_batchs = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(validation_batchs//2)
validation_dataset = validation_dataset.skip(validation_batchs//2)

print("Number of validation batches: {}".format(tf.data.experimental.cardinality(validation_dataset)))
print("Number of test batches: {}".format(tf.data.experimental.cardinality(test_dataset)))

# preprocess data
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(600).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# create augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# create preprocess layer
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# create normalization layer
normalization_layer = Rescaling(1./255)

# load model
base_model = tf.keras.applications.ResNet50(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
base_model.trainable=False

# create global_average layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# create prediction layer
prediction_layer = tf.keras.layers.Dense(8)

# build the model
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
X = normalization_layer(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# first evaluation
model.evaluate(test_dataset)

# first training
initial_epochs = 10
history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)

# unfreeze the model partialy
base_model.trainable = True
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=["accuracy"])

# second training
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset)

# plotting training results
acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0, 0.6])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([1.0, 4.0])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

# post training evaluation
model.evaluate(test_dataset)

# export weights
output_model = Model(inputs=base_model.inputs, outputs=model.layers[-4].output)
output_model.save_weights("{}Checkpoint-{}".format(OUTPUT_PATH, time.strftime("%d,%m,%Y-%H,%M,%S")))