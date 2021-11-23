from torch.utils.data import Dataset
import json
import os
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, data_dir, part, mode, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.part = part
        self.mode = mode
        self.transform = transform

        patch_size = {"left_eye": 128,
                      "right_eye": 128, "mouth": 192, "nose": 160, "face": 512}
        self.patch_size = patch_size[part]
        self.image_paths = []
        self.points = []

        self.setup(part, mode)

    def __getitem__(self, index):

        image = self.read_image(index)
        point = list(map(int, self.points[index]))
        if self.part != "face":
            patch_image = image[point[0]-(self.patch_size)//2:point[0] +
                                (self.patch_size)//2, point[1]-(self.patch_size)//2:point[1]+(self.patch_size)//2]

        else:
            image[point[0][0]-64:point[0][0]+64,
                  point[0][1]-64:point[0][1]+64] = 0
            image[point[1][0]-64:point[0][0]+64,
                  point[1][1]-64:point[0][1]+64] = 0
            image[point[2][0]-96:point[0][0]+96,
                  point[2][1]-96:point[0][1]+96] = 0
            image[point[3][0]-80:point[0][0]+80,
                  point[3][1]-80:point[0][1]+80] = 0

            patch_image = image
            patch_image_trans = patch_image

        if self.transform is not None:
            patch_image_trans = self.transform(patch_image)

        return patch_image, patch_image_trans

    def __len__(self):
        return len(self.image_paths)

    def setup(self, part, mode):
        json_path = os.path.join(self.data_dir, f"{mode}.json")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        for img in json_data:
            image_path = os.path.join(self.data_dir, img)
            if part != "face":
                image_part_point = json_data[img][part]
            else:
                image_part_point = []
                image_part_point.append(json_data[img]["left_eye"])
                image_part_point.append(json_data[img]["right_eye"])
                image_part_point.append(json_data[img]["mouth"])
                image_part_point.append(json_data[img]["nose"])
                image_part_point.append([256, 256])

            self.image_paths.append(image_path)
            self.points.append(image_part_point)

    def read_image(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
        image /= 256
        return image
