import os
import cv2
import json

import numpy as np
root_folder='/home/myamya/project/image_files/'
info_path = '/home/myamya/project/image_files/annotationsx4/annotationsx4_file.json'
save_path = '/home/myamya/project/image_files/alignment_real'

info = json.load(open(info_path, 'r'))
print(info)
 

for info_key in info:

    cur_info = info[info_key]
    img_path    = os.path.join(root_folder,'cropped_512_image' , info_key)
    sketch_path = os.path.join(root_folder,'sketched2',info_key)
    print(img_path)
    print(sketch_path)
    img         = cv2.imread(img_path)

    sketch      = cv2.imread(sketch_path)

    img_idx     = img_path.split('.')[-2].split('/')[-1]

    print(img_path, img_idx)

 

    part = ['left_eye', 'right_eye', 'nose', 'mouth', 'face_fit']

    sz   = [128, 128, 160, 192, 512]
 
    cur_info['face_fit'] = [[cur_info['face_fit'][0]],[cur_info['face_fit'][2]]]
	
    pos  = [cur_info[part[i]] for i in range(5)]; pos[-1] = [256, 256]
    pos  = [pos[0], pos[1],pos[3]]
#    pos = [[155,254],[322,237],[246,377]]
    dst  = [[186, 244], [327, 244], [256, 385]]

    pos = np.float32(pos)

    dst = np.float32(dst)

    M   = cv2.getAffineTransform(pos,dst)
    img    = cv2.warpAffine(img, M, (512, 512), borderValue=(255, 255, 255))

    sketch = cv2.warpAffine(sketch, M, (512, 512), borderValue=(255, 255, 255))

    cv2.imwrite(os.path.join(root_folder, 'alignment_real', img_idx+'.png'), img)

    cv2.imwrite(os.path.join(root_folder, 'alignment_sketch', img_idx+'.png'), sketch)

