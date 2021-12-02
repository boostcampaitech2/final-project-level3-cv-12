from torch.utils.data import Dataset
import json
import os
import cv2
from module_fold import CEModule
import torch


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
        point = self.points[index]

        if self.part != "face":
            xmin = point[0]-(self.patch_size)//2
            xmax = point[0] + (self.patch_size)//2
            ymin = point[1]-(self.patch_size)//2
            ymax = point[1]+(self.patch_size)//2

            if xmin < 0:
                xmin, xmax = 0, self.patch_size
            if xmax > 512:
                xmin, xmax = 512 - self.patch_size, 512
            if ymin < 0:
                ymin, ymax = 0, self.patch_size
            if ymax > 512:
                ymin, ymax = 512 - self.patch_size, 512

            patch_image = image[ymin:ymax, xmin:xmax]
            patch_image_trans = patch_image

        else:
            image[point[0][1]-64:point[0][1]+64,
                  point[0][0]-64:point[0][0]+64] = 1
            image[point[1][1]-64:point[1][1]+64,
                  point[1][0]-64:point[1][0]+64] = 1
            image[point[2][1]-96:point[2][1]+96,
                  point[2][0]-96:point[2][0]+96] = 1
            image[point[3][1]-80:point[3][1]+80,
                  point[3][0]-80:point[3][0]+80] = 1

            patch_image = image
            patch_image_trans = patch_image

        if self.transform is not None:
            patch_image_trans = self.transform(image=patch_image)
            patch_image_trans = patch_image_trans["image"]

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
                image_part_point = list(map(int, json_data[img][part]))
            else:
                image_part_point = []
                image_part_point.append(
                    list(map(int, json_data[img]["left_eye"])))
                image_part_point.append(
                    list(map(int, json_data[img]["right_eye"])))
                image_part_point.append(
                    list(map(int, json_data[img]["mouth"])))
                image_part_point.append(list(map(int, json_data[img]["nose"])))

            self.image_paths.append(image_path)
            self.points.append(image_part_point)

    def read_image(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
        image /= 255
        return image


class CustomVectorset(Dataset):
    def __init__(self, img_dir, sketch_dir, pth_dir, mode):
        super().__init__()
        self.img_dir = img_dir
        self.sketch_dir = sketch_dir
        self.pth_dir = pth_dir
        self.part = ['mouth', 'left_eye', 'right_eye', 'nose', 'face']
        self.mode = mode

    def __getitem__(self, index):

        part_encoder = {}

        mouth_v = part_encoder["mouth"](mouth_patch)
        l_eye_v = part_encoder["left_eye"](l_eye_patch)
        r_eye_v = part_encoder["right_eye"](r_eye_patch)
        nose_v = part_encoder["nose"](nose_patch)
        face_v = part_encoder["face"](face_patch)

        real_img =
        return mouth_v, l_eye_v, r_eye_v, nose_v, face_v, real_img, points

    def __len__(self):
        return
