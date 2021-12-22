import cv2 
import os 
def dodge_img(x,y): 
    return cv2.divide(x,255-y,scale=256)

def burn_img(image, mask): 
    temp = cv2.divide(255-image, 255-mask, scale=256)
    return (255.-temp)


def sketch(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)
    gblur = cv2.GaussianBlur(invert,(27,27), sigmaX=0, sigmaY=0)
    dodged=dodge_img(gray,gblur)
    final=burn_img(dodged,gblur)
    return final


base_path = '/home/myamya/project/image_files/cropped_512_image'
dest_path = '/home/myamya/project/image_files/sketched1'
images = os.listdir(base_path)
for i in images:
    final = sketch(os.path.join(base_path, i))
    cv2.imwrite(os.path.join(dest_path,i), final)

