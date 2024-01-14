#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import os


class CityScapes(Dataset):
    def __init__(self, mode, image_dir=Path('/content/Cityscapes/Cityspaces/'), im_size=(1024,512)):
        super(CityScapes, self).__init__()
        self.mode = mode
        self.im_size = im_size

        # get images and ground truths directories
        self.images = []
        self.image_dir = os.path.join(image_dir, 'images', mode)
        self.ground_truth_dir = self.image_dir.replace('images', 'gtFine')
        folders = os.listdir(self.image_dir)
        # get all the images
        for folder in folders:
            folder_path = os.path.join(self.image_dir, folder)
            image_names = os.listdir(folder_path)
            self.images.extend(image_names)

        # create a transform to convert image to tensor
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, idx):
      # get image path
      img_folder = self.images[idx].split('_')[0]
      img_path = os.path.join(self.image_dir, img_folder, self.images[idx])
      # create PIL image
      image = Image.open(img_path).convert('RGB')
      # resize image - use BILINEAR to get a smoother image
      image = image.resize(self.im_size, Image.BILINEAR)
      # convert image to tensor
      image = self.to_tensor(image)

      # get ground truth path
      ground_truth_path = os.path.join(self.ground_truth_dir, img_folder, self.images[idx].replace('leftImg8bit', 'gtFine_labelTrainIds'))
      # create PIL image of ground truth
      ground_truth = Image.open(ground_truth_path)
      # resize ground truth - use NEAREST to keep the values of the classes intact
      ground_truth = ground_truth.resize(self.im_size, Image.NEAREST)
      # convert ground truth to numpy array - add an extra dimension with np.newaxis to match image shape
      ground_truth = np.array(ground_truth).astype(np.int64)[np.newaxis, :]

      return image, ground_truth

    def __len__(self):
        return len(self.images)
