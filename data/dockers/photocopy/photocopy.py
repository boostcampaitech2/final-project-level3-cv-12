import cv2
import matplotlib.pyplot as plt
import numpy as np

def pencilSketch(image):
    # Converting image to gray-scale
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Gaussian blurring to remove noise with a kernel-size of 5
    img_g=cv2.GaussianBlur(image,(5,5),0,0)
    # Calculating the laplacian of the image.
    # Notice how the 2nd parameter changes the data-type of the image to float32
    # Kernel size used is 3 and scale and delta values are determined by experimentations
    img_l=cv2.Laplacian(img_g,cv2.CV_32F,3,scale=25,delta=50)
    # Min-Max Normalizing the image to perform further operations
    cv2.normalize(img_l,dst=img_l,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F);
    # Converting the image back to uint8 format
    img_i=np.uint8(img_l*255)
    # Thresholding the image to remove any noise (in this case tiny spots of pencil marks all around the image)
    img_t=cv2.threshold(img_i,147,255,cv2.THRESH_BINARY_INV)[1]
    # Converting in back to 3 color channel for display
    pencilSketchImage=cv2.cvtColor(img_t,cv2.COLOR_GRAY2BGR)
    return pencilSketchImage

imagePath= '/home/myamya/project/image_files/annotationsx4/0.jpg'
img = cv2.imread(imagePath)
# Reading the image using openCV
pencil_image = pencilSketch(img)
cv2.imwrite('/home/myamya/project/image_files/annotationsx4/0_sketch.jpg',pencil_image)


