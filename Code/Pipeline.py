import cv2 as cv

img = cv.imread("~/PowerFolders/Masterarbeit/Bilder/acht/acht1_001.jpg")

while cv.waitKey(1) != ord('x'):
    cv.imshow("Bild", img)