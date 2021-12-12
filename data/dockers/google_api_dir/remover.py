import os
from PIL import Image
import numpy as np
from PIL import ExifTags
import piexif

not_orient=[]
not_exif=[]

def rotate(path, filename):


    img = Image.open(os.path.join(path,filename))

    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])
        #print(exif_dict)
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            try:

               exif_bytes = piexif.dump(exif_dict)
            except:
                print('wrong file delete')
                os.remove(os.path.join(path,filename))
                return 0
            print('{} orientation value is {}'.format(filename,str(orientation)))
            
            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 3:
                img = img.rotate(180)

            elif orientation == 4:
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 6:
                img = img.rotate(-90, expand=True)

            elif orientation == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

            elif orientation == 8:
                img = img.rotate(90, expand=True)

            img.save(os.path.join(path,filename), exif=exif_bytes)
        else:
            #print('{} doesn\'t have exif orient info.'.format(filename))
            not_orient.append(filename)

    else:
        not_exif.append(filename)

path = "/home/myamya/project/image_files/images"
images = os.listdir(path)
for i in images:
    rotate(path, i)

