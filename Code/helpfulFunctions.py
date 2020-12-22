import cv2 as cv
import numpy as np
import os
import shutil

try:
    def showImage(img):
        cv.imshow("Bild", np.array(img, dtype=np.uint8))
        cv.waitKey(15000)
        cv.destroyWindow("Bild")
    
    def saveImage(img, output_path):
        cv.imwrite(output_path + "prediction.jpg", img)
    
    def splitTrainTest(path, split):
        if os.path.isdir(path + "train_split") == False:
            os.makedirs(path + "train_split")
        if os.path.isdir(path + "test_split") == False:
            os.makedirs(path + "test_split")

        images = sorted([element for element in os.listdir(path) if (element.endswith(".JPG")) or (element.endswith(".jpg"))])
        jsons = sorted([element for element in os.listdir(path) if element.endswith(".json")])

        for count in range(100):
            if count % split == 0:
                shutil.move(path + images[count], path + "test_split/" + images[count])
                shutil.move(path + jsons[count], path + "test_split/" + jsons[count])
            else:
                shutil.move(path + images[count], path + "train_split/" + images[count])
                shutil.move(path + jsons[count], path + "train_split/" + jsons[count])

finally:
    print("")
