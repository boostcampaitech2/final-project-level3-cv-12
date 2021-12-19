import json
import cv2
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    part_size = { 'left_eye': 128, 'right_eye': 128, 'nose': 168, 'mouth': 192, 'remainder': 512 }

    def __init__(self, json_path):
        self.info = json.load(open(json_path, 'r'))
    
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_center(self, idx, part):
        x, y = self.info[str(idx)][part]
        x, y = int(x), int(y)
        sz   = self.part_size[part] // 2
        if x - sz < 0:   x = sz
        if x + sz > 512: x = 512 - sz
        if y - sz < 0:   y = sz
        if y + sz > 512: y = 512 - sz
        return x, y


class CEDataset(BaseDataset):
    def __init__(self, json_path, part, transform=None, transform_all=None):
        super().__init__(json_path)
        self.part = part # 'left_eye', 'right_eye', 'nose', 'mouth', 'remainder', 'all'
        self.transform     = transform
        self.transform_all = transform_all

    def __getitem__(self, idx):
        # load image
        img_path = self.info[str(idx)]['sketch_path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img = 1 - img / 255

        # crop
        if self.part != 'remainder' and self.part != 'all':
            x, y = self._get_center(idx, self.part)
            sz   = self.part_size[self.part] // 2
            img  = img[y-sz:y+sz, x-sz:x+sz]
        elif self.part != 'all':
            # x1, y1 = self._get_center(idx, 'left_eye')
            # x2, y2 = self._get_center(idx, 'right_eye')
            # x3, y3 = self._get_center(idx, 'nose')
            # x4, y4 = self._get_center(idx, 'mouth')
            x1, y1 = 186, 244
            x2, y2 = 326, 244
            x3, y3 = 256, 302
            x4, y4 = 256, 385
            sz1 = self.part_size['left_eye'] // 2
            sz2 = self.part_size['right_eye'] // 2
            sz3 = self.part_size['nose'] // 2 - 10
            sz4 = self.part_size['mouth'] // 2 - 15
            img[y1-sz1:y1+sz1, x1-sz1:x1+sz1] = 0
            img[y2-sz2:y2+sz2, x2-sz2:x2+sz2] = 0
            img[y3-sz3:y3+sz3, x3-sz3:x3+sz3] = 0
            img[y4-sz4:y4+sz4, x4-sz4:x4+sz4] = 0

        # apply transform (Denoising AE)
        img_trans = img
        if self.transform is not None:
            img_trans = self.transform(image=img)['image']
        if self.transform_all is not None:
            img = self.transform_all(image=img, image_trans=img_trans)
            img, img_trans = img['image'], img['image_trans']
        return img, img_trans


class FEDataset(BaseDataset):
    def __init__(self, json_path):
        super().__init__(json_path)
    
    def __len__(self, ):
        return len(self.info)
    
    def __getitem__(self, idx):
        img_path = self.info[str(idx)]['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(float)
        img = img / 255

        sketch_path = self.info[str(idx)]['sketch_path']
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE).astype(float)
        sketch = 1 - sketch / 255

        x1, y1 = self._get_center(idx, 'left_eye')
        x2, y2 = self._get_center(idx, 'right_eye')
        x3, y3 = self._get_center(idx, 'nose')
        x4, y4 = self._get_center(idx, 'mouth')
        x5, y5 = 256, 256
        point = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]

        return img, sketch, point
