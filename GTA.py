#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import random
import os


class CityScapes(Dataset):
    def __init__(self, mode, image_dir=Path('/content/GTA5/'), im_size=(1024,512)):
        super(CityScapes, self).__init__()
        self.mode = mode
        self.im_size = im_size

        self.images = []
        self.image_dir = os.path.join(image_dir, 'images', mode)
        self.ground_truth_dir = self.image_dir.replace('images', 'labels')
        image_names = os.listdir(self.image_dir)
        self.images.extend(image_names)

        self.to_tensor = transforms.ToTensor()
        self.label_map = self.get_label_map()
        self.p = 0.5


    def convert_labels(self, ground_truth):
        converted_label = np.zeros((*np.array(ground_truth).shape[:-1], 1), dtype=np.int64)

        for label in self.label_map:
            color_array = np.array(label['color'])
            mask = np.all(np.array(ground_truth) == color_array, axis=-1, keepdims=True)
            converted_label[mask] = label['ID']
        converted_label = np.transpose(converted_label, (2, 0, 1))
        return converted_label

    def __getitem__(self, idx):
      img_path = os.path.join(self.image_dir, self.images[idx])
      ground_truth_path = os.path.join(self.ground_truth_dir, self.images[idx])
      image = Image.open(img_path).convert('RGB')
      image = image.resize(self.im_size, Image.BILINEAR)
      image = self.to_tensor(image)

      ground_truth = Image.open(ground_truth_path).convert('RGB')
      ground_truth = ground_truth.resize(self.im_size, Image.BILINEAR)
      ground_truth = np.array(ground_truth).astype(np.int64)
      ground_truth = self.convert_labels(ground_truth)

      return image, ground_truth

    def __len__(self):
        return len(self.images)

    def get_label_map(self):
      road = {'ID':0, 'color':(128, 64, 128)}
      sidewalk = {'ID':1, 'color':(244, 35, 232)}
      building = {'ID':2, 'color':(70, 70, 70)}
      wall = {'ID':3, 'color':(102, 102, 156)}
      fence = {'ID':4, 'color':(190, 153, 153)}
      pole = {'ID':5, 'color':(153, 153, 153)}
      light = {'ID':6, 'color':(250, 170, 30)}
      sign = {'ID':7, 'color':(220, 220, 0)}
      vegetation = {'ID':8, 'color':(107, 142, 35)}
      terrain = {'ID':9, 'color':(152, 251, 152)}
      sky = {'ID':10, 'color':(70, 130, 180)}
      person = {'ID':11, 'color':(220, 20, 60)}
      rider = {'ID':12, 'color':(255, 0, 0)}
      car = {'ID':13, 'color':(0, 0, 142)}
      truck = {'ID':14, 'color':(0, 0, 70)}
      bus = {'ID':15, 'color':(0, 60, 100)}
      train = {'ID':16, 'color':(0, 80, 100)}
      motorcycle = {'ID':17, 'color':(0, 0, 230)}
      bicycle = {'ID':18, 'color':(119, 11, 32)}

      label_map = [
          road,
          sidewalk,
          building,
          wall,
          fence,
          pole,
          light,
          sign,
          vegetation,
          terrain,
          sky,
          person,
          rider,
          car,
          truck,
          bus,
          train,
          motorcycle,
          bicycle,
      ]
      return label_map