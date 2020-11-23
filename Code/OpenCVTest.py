import cv2

img = cv2.imread("/home/julius/PowerFolders/Masterarbeit/Bilder/acht/acht1_001.jpg")

img_b = img[:,:,0]
img_r = img[:,:,1]
img_g = img[:,:,2]

while cv2.waitKey(1) != ord('x'):
    cv2.imshow("Bild Blau", img_b)
    cv2.imshow("Bild Rot", img_r)
    cv2.imshow("Bild Grün", img_g)