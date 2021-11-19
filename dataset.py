from torch.utils.data import Dataset
import json
import os
from PIL import Image


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

        self.setup(part)

    def __getitem__(self, index):

        image = self.read_image(index)
        points = self.points[index]
        if self.part != "face":
            patch_image = image[points[0]-(self.patch_size)//2:points[0] +
                                (self.patch_size)//2, points[1]-(self.patch_size)//2:points[1]+(self.patch_size)//2]

        else:
            image[points[0][0]-64:points[0][0]+64,
                  points[0][1]-64:points[0][1]+64] = 0
            image[points[1][0]-64:points[0][0]+64,
                  points[1][1]-64:points[0][1]+64] = 0
            image[points[2][0]-96:points[0][0]+96,
                  points[2][1]-96:points[0][1]+96] = 0
            image[points[3][0]-80:points[0][0]+80,
                  points[3][1]-80:points[0][1]+80] = 0

            patch_image = image
            patch_image_trans = patch_image

        if self.transform is not None:
            patch_image_trans = self.transform(patch_image)

        return patch_image, patch_image_trans

    def setup(self, part, mode):
        json_path = os.path.join(self.data_dir, f"{mode}.json")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        image_list = json_data["file_name"]
        for img in image_list:
            image_path = os.path.join(self.data_dir, img)
            if part != "face":
                image_part_point = json_data[img][part]
            else:
                image_part_point = []
                image_part_point.add(json_data[img]["left_eye"])
                image_part_point.add(json_data[img]["right_eye"])
                image_part_point.add(json_data[img]["mouth"])
                image_part_point.add(json_data[img]["nose"])
                image_part_point = [256, 256]

            self.image_paths.append(image_path)
            self.points.append(image_part_point)

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.oepn(image_path).convert("RGB")
