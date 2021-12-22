import json
import cv2
from torch.utils.data import Dataset
import numpy as np
import os


class CEDataset(Dataset):
    pos = {'left_eye': (244, 186), 'right_eye': (244, 326), 'nose': (
        302, 256), 'mouth': (385, 256), 'remainder': (256, 256), 'all': (256, 256)}
    sz = {'left_eye': 128, 'right_eye': 128, 'nose': 160,
          'mouth': 192, 'remainder': 512, 'all': 512}

    def __init__(self, json_path, part, transform=None, transform_all=None):
        self.info = json.load(open(json_path, 'r'))
        self.part = part
        self.transform = transform
        self.transform_all = transform_all

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        # load image
        img_path = self.info[str(idx)]['sketch']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img = 1 - img / 255

        # crop
        if self.part != 'remainder' and self.part != 'all':
            x, y = self.pos[self.part]
            sz = self.sz[self.part] // 2
            img = img[x-sz:x+sz, y-sz:y+sz]
        elif self.part != 'all':
            x1, y1 = self.pos['left_eye']
            x2, y2 = self.pos['right_eye']
            x3, y3 = self.pos['nose']
            x4, y4 = self.pos['mouth']
            sz1 = self.sz['left_eye'] // 2
            sz2 = self.sz['right_eye'] // 2
            sz3 = self.sz['nose'] // 2
            sz4 = self.sz['mouth'] // 2
            img[x1-sz1:x1+sz1, y1-sz1:y1+sz1] = 0
            img[x2-sz2:x2+sz2, y2-sz2:y2+sz2] = 0
            img[x3-sz3:x3+sz3, y3-sz3:y3+sz3] = 0
            img[x4-sz4:x4+sz4, y4-sz4:y4+sz4] = 0

        # apply transform
        img_trans = img
        if self.transform is not None:
            img_trans = self.transform(image=img)['image']
        if self.transform_all is not None:
            img = self.transform_all(image=img, image_trans=img_trans)
            img, img_trans = img['image'], img['image_trans']
        return img, img_trans


class FEDataset(Dataset):
    def __init__(self, patch_path, image_json_path, transform=None):
        super().__init__()
        self.image_json_path = image_json_path
        self.patch_path = patch_path
        self.transform = transform
        self.info = json.load(open(self.image_json_path, 'r'))

    def __len__(self, ):
        return len(self.info)

    def __getitem__(self, idx):
        pos = {'remainder': (256, 256), 'nose': (302, 256), 'right_eye': (
            244, 326), 'left_eye': (244, 186), 'mouth': (385, 256)}
        sz = {'remainder': 512, 'nose': 160,
              'right_eye': 128, 'left_eye': 128, 'mouth': 128}

        img_path = self.info[str(idx)]['image']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(float)
        img = img / 255

        output = np.zeros((5, 512, 512))
        for i, part in enumerate(pos.keys()):
            patch_path = os.path.join(self.patch_path, part, str(idx)+".png")
            patch_img = cv2.imread(
                patch_path, cv2.IMREAD_GRAYSCALE).astype(float)

            if self.transform is not None:
                patch_img = self.transform(image=patch_img)['image']

            output[i, pos[part][0]-sz[part]//2:pos[part][0]+sz[part]//2,
                   pos[part][1]-sz[part]//2:pos[part][1]+sz[part]//2] = patch_img
        return output, img
