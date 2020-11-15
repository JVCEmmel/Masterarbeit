from PIL import Image, ImageFilter
import numpy as np
import pytesseract
import cv2

"""
A function which devides Pillow Images in their three color channels
Therefore the image is converted to numpy and back

The function needs a pillow image

The funciton returns three pillow images (RGB)
"""

def pic_split(img):

    #converting into numpy
    img_array = np.array(img).astype(np.float)
    
    print(img_array.shape)

    #the actual split
    img_array_red = img_array[:,:,0]
    img_array_green = img_array[:,:,1]
    img_array_blue = img_array[:,:,2]

    np.where(img_array_red<128, 1, 0)
    np.where(img_array_green<128, 1, 0)
    np.where(img_array_blue<128,1 ,0)
    print(img_array_red.shape)
    print(img_array_green.shape)
    print(img_array_blue.shape)

    #converting them back
    img_red = Image.fromarray(img_array_red.astype(np.uint8))
    img_green = Image.fromarray(img_array_green.astype(np.uint8))
    img_blue = Image.fromarray(img_array_blue.astype(np.uint8))

    return img_red, img_green, img_blue

img = Image.open("/home/julius/PowerFolders/Masterarbeit/Bilder/acht/acht1_003.jpg")

print(img.format, img.size, img.mode)

img_red, img_green, img_blue = pic_split(img)

#print the format of the three color channel images

print(img_red.format)
print(img_green.format)
print(img_blue.format)


"""
img_red = img_red.filter(ImageFilter.MinFilter(3))
img_green = img_green.filter(ImageFilter.MinFilter(3))
img_blue = img_blue.filter(ImageFilter.MinFilter(3))
"""

#show the three color channel images

#img_red.show()
#img_green.show()
img_blue.show()


#grayscaling the image, printing its form and show
"""
img = img.convert('1')
img = img.filter(ImageFilter.MinFilter(3))

print(img.format, img.size, img.mode)

img.show()
"""
print("Red\n" + str(pytesseract.image_to_string(img_red)))
#print("Green\n" + str(pytesseract.image_to_string(img_green)))
#print("Blue\n" + str(pytesseract.image_to_string(img_blue)))
