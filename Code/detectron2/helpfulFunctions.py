import cv2 as cv
import numpy as np
import os
import shutil

try:
    def showImage(img):
        cv.imshow("Bild", np.array(img, dtype=np.uint8))
        cv.waitKey(15000)
        cv.destroyWindow("Bild")

finally:
    print("")
