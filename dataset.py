import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import cv2
import numpy as np
# from data_loader.aug import InitTransform


class BaseDataset(Dataset):
    part_size = {'left_eye': 128, 'right_eye': 128,
                 'nose': 168, 'mouth': 192, 'remainder': 512}

    def __init__(self, json_path):
        self.info = json.load(open(json_path, 'r'))

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_center(self, idx, part):
        x, y = self.info[str(idx)][part]
        x, y = int(x), int(y)
        sz = self.part_size[part] // 2
        if x - sz < 0:
            x = sz
        if x + sz > 512:
            x = 512 - sz
        if y - sz < 0:
            y = sz
        if y + sz > 512:
            y = 512 - sz
        return x, y


class CEDataset(BaseDataset):
    def __init__(self, json_path, part, transform=None):
        super().__init__(json_path)
        self.part = part  # 'left_eye', 'right_eye', 'nose', 'mouth', 'remainder'
        # self.init_transform = InitTransform(512)
        self.transform = transform

    def __getitem__(self, idx):
        # load image
        img_path = self.info[str(idx)]['sketch_path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        # img      = self.init_transform(img)
        img = img / 255
        img = np.ones(img.shape(), dtype=np.float32) - img

        # crop
        if self.part != 'remainder':
            x, y = self._get_center(idx, self.part)
            sz = self.part_size[self.part] // 2
            img = img[y-sz:y+sz, x-sz:x+sz]
        else:
            x1, y1, sz1 = self._get_center(
                'left_eye'), self.part_size['left_eye'] // 2
            x2, y2, sz2 = self._get_center(
                'right_eye'), self.part_size['right_eye'] // 2
            x3, y3, sz3 = self._get_center('nose'), self.part_size['nose'] // 2
            x4, y4, sz4 = self._get_center(
                'mouth'), self.part_size['mouth'] // 2
            img[y1-sz1:y1+sz1, x1-sz1:x1+sz1] = 0
            img[y2-sz2:y2+sz2, x2-sz2:x2+sz2] = 0
            img[y3-sz3:y3+sz3, x3-sz3:x3+sz3] = 0
            img[y4-sz4:y4+sz4, x4-sz4:x4+sz4] = 0

        # apply transform (Denoising AE)
        img_trans = img
        if self.transform is not None:
            img_trans = self.transform(img)
        return img, img_trans


class FEDataset(BaseDataset):
    def __init__(self, json_path, fv_path):
        super().__init__(json_path)
        self.fv = json.load(open(fv_path, 'r'))
        # self.init_transform = InitTransform(512)

    def __len__(self, ):
        return len(self.info)

    def __getitem__(self, idx):  # img(1), point(5), fv(5)
        img_path = self.info[str(idx)]['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(float)
        # img = self.init_transform(img)
        img = img / 255

        x1, y1 = self._get_center(idx, 'left_eye')
        x2, y2 = self._get_center(idx, 'right_eye')
        x3, y3 = self._get_center(idx, 'nose')
        x4, y4 = self._get_center(idx, 'mouth')
        point = torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        fv1 = torch.tensor(self.fv[str(idx)]['left_eye'])
        fv2 = torch.tensor(self.fv[str(idx)]['right_eye'])
        fv3 = torch.tensor(self.fv[str(idx)]['nose'])
        fv4 = torch.tensor(self.fv[str(idx)]['mouth'])
        fv5 = torch.tensor(self.fv[str(idx)]['remainder'])
        fv = torch.stack([fv1, fv2, fv3, fv4, fv5], dim=0)

        return img, point, fv
