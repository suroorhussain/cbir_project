import cv2
import mahotas
import numpy as np

class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        
    def describe(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Create a border around the image
        image = cv2.copyMakeBorder(image, 15, 15, 15, 15,
		cv2.BORDER_CONSTANT, value = 255)

	# invert the image and threshold it
	thresh = cv2.bitwise_not(image)
	thresh[thresh > 0] = 255

        #Create outlines of the shape draw only the largest 3 areas
        outline = np.zeros(image.shape, dtype = "uint8")
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
	cv2.drawContours(outline, cnts, -1, 255, -1)

        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(outline, self.radius)
