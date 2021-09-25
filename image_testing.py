from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

model = keras.models.load_model('model.h5')

#loading image from tests folder and loading it in as grayscale
img = cv2.imread('./tests/test9.jpg', 0)
plt.imshow(img)
plt.show()

#resizing image to a 28*28 picture and reversing the colors.
#Interpolation is necessary here because without it too much of the number is lost
img = cv2.resize(255-img, (28,28), interpolation=cv2.INTER_AREA)
#plt.imshow(img)
#plt.show()

#Getting rid of any irregularities in the image so that only the number and a dark blackground remain
#TODO remove glare from image in some images the glare gets picked up as part of the number and this ruins the image
(thresh, img) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#plt.imshow(img)
#plt.show()

#This next whole part centers the image
#The images are fit in a 20x20 pixel box and are centered in a 28x28 image using the center of mass
#Every row and column at the sides of the image are removed which are completely black
while np.sum(img[0]) == 0:
    img = img[1:]
while np.sum(img[:,0]) == 0:
    img = np.delete(img,0,1)
while np.sum(img[-1]) == 0:
    img = img[:-1]
while np.sum(img[:,-1]) == 0:
    img = np.delete(img, -1,1)

rows, cols = img.shape

#images are resized and fit into a 20x20 box
if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    img = cv2.resize(img, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    img = cv2.resize(img, (cols,rows))

#Adding the missing rows and columns with the np.lib.pad function to get a 28x28 picture
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
img = np.lib.pad(img,(rowsPadding,colsPadding), 'constant')

#getting the center of mass
shiftx,shifty = getBestShift(img)
#shifting the image in the given directions
shifted = shift(img,shiftx,shifty)
img = shifted
#plt.imshow(img)
#plt.show()

#making sure the image can have decimal points
img = img.astype('float32')
#Same as in the model reshape the image and divide it by 255
img = img.reshape(1, 28, 28, 1)
img /= 255

pred = model.predict(img)
print(pred.argmax())