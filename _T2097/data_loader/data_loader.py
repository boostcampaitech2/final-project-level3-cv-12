from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json, cv2, os
from aug import InitTransform


class CEDataset(Dataset):
    part_size = { "left_eye": 128, "right_eye": 128, "nose": 168, "mouth": 192, "remainder": 512 }

    def __init__(self, path, part, transform=None):
        self.path = path # '~~/data/'
        self.part = part # "left_eye", "right_eye", "nose", "mouth", "remainder"
        self.info = json.load(open(path + 'info.json', 'r'))
        self.init_transform = InitTransform(self.part_size[part])
        self.transform = transform
    
    def __len__(self, ):
        return len(self.info)

    def __getitem__(self, idx):
        # load image
        img_path = self.info[str(idx)]['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)

        # crop
        if self.part != "remainder":
            pass
        else:
            pass
        img = self.init_transform(img)

        # apply transform (Denoising AE)
        img_trans = img
        if self.transform is not None:
            img_trans = self.transform(img)
        return img, img_trans


class FEDataset(Dataset):
    def __init__(self, path):
        pass
    
    def __len__(self, ):
        pass
    
    def __getitem__(self, idx):
        pass