class Transform:
    import numpy as np
    import math
    import cv2
    from scipy import ndimage

    def __init__(self):
        # Nothing needs to be done except get the libraries and functions ready
        pass

    def getBestShift(self, img):
        # TODO comments
        cy,cx = self.ndimage.measurements.center_of_mass(img)
        rows,cols = img.shape
        shiftx = self.np.round(cols/2.0-cx).astype(int)
        shifty = self.np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty

    def shift(self, img, sx, sy):
        # TODO Comments
        rows,cols = img.shape
        M = self.np.float32([[1,0,sx],[0,1,sy]])
        shifted = self.cv2.warpAffine(img,M,(cols,rows))
        return shifted

    def center(self, img):
        # TODO Comments
        while self.np.sum(img[0]) == 0:
            img = img[1:]
        while self.np.sum(img[:,0]) == 0:
            img = self.np.delete(img,0,1)
        while self.np.sum(img[-1]) == 0:
            img = img[:-1]
        while self.np.sum(img[:,-1]) == 0:
            img = self.np.delete(img, -1,1)
        return img

    def numberTransform(self, img):
        # TODO Comments for most steps, not everything is clear
        img = self.cv2.resize(255-img, (28,28), interpolation=self.cv2.INTER_AREA)
        (thresh, img) = self.cv2.threshold(img, 130, 255, self.cv2.THRESH_BINARY | self.cv2.THRESH_OTSU)
        img = self.center(img)
        rows, cols = img.shape

        #images are resized and fit into a 20x20 box
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            img = self.cv2.resize(img, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            img = self.cv2.resize(img, (cols,rows))

        #Adding the missing rows and columns with the np.lib.pad function to get a 28x28 picture
        colsPadding = (int(self.math.ceil((28-cols)/2.0)),int(self.math.floor((28-cols)/2.0)))
        rowsPadding = (int(self.math.ceil((28-rows)/2.0)),int(self.math.floor((28-rows)/2.0)))
        img = self.np.lib.pad(img,(rowsPadding,colsPadding), 'constant')

        #getting the center of mass
        shiftx,shifty = self.getBestShift(img)
        #shifting the image in the given directions
        shifted = self.shift(img,shiftx,shifty)
        img = shifted

        img = img.astype('float32')
        #Same as in the model reshape the image and divide it by 255
        img = img.reshape(1, 28, 28, 1)
        img /= 255

        # Image has been prepared!
        return img