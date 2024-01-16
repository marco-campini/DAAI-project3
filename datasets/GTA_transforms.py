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
        # define a transform to apply to the image
        self.img_transform = transforms.Compose([
          transforms.RandomHorizontalFlip(p=1),
          transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7)
        ])
        # define a transform to apply to the ground truth
        self.gt_transform = transforms.RandomHorizontalFlip(p=1)
        self.im_size = im_size

        # get all the images
        self.images = []
        self.image_dir = os.path.join(image_dir, 'images', mode)
        self.ground_truth_dir = self.image_dir.replace('images', 'labels')
        image_names = os.listdir(self.image_dir)
        self.images.extend(image_names)

        # create a transform to convert image to tensor
        self.to_tensor = transforms.ToTensor()
        # get the map with the ground truth labels
        self.label_map = self.__get_label_map__()

    # define a method that converts a certain color to its corresponding label
    def __convert_labels__(self, ground_truth):
        # create a numpy array the same size of the ground truth and put zero everywhere
        converted_label = np.zeros((*ground_truth.shape[:-1], 1), dtype=np.int64)
        # check each label in the label map
        for label in self.label_map:
            # create a mask that equals 'True' where the color of the current label is found
            color_array = np.array(label['color'])
            mask = np.all(ground_truth == color_array, axis=-1, keepdims=True)
            # populate the numpy array with the label id using the mask
            converted_label[mask] = label['ID']
        # transpose the array to match the shape of the image
        converted_label = np.transpose(converted_label, (2, 0, 1))
        return converted_label

    def __getitem__(self, idx):
        # get image path
        img_path = os.path.join(self.image_dir, self.images[idx])
        # create PIL image
        image = Image.open(img_path).convert('RGB')
        # resize image - use BILINEAR to get a smoother image
        image = image.resize(self.im_size, Image.BILINEAR)
        # convert image to tensor
        image = self.to_tensor(image)

        # get ground truth path
        ground_truth_path = os.path.join(self.ground_truth_dir, self.images[idx])
        # create PIL image of ground truth
        ground_truth = Image.open(ground_truth_path).convert('RGB')
        # resize ground truth - use BILINEAR to get a smoother image, no need to keep the labels intact
        ground_truth = ground_truth.resize(self.im_size, Image.BILINEAR)
        # apply the transforms with a 50% probability
        if random.random() > 0.5 and mode=='train':
          image = self.img_transform(image)
          ground_truth = self.gt_transform(ground_truth)
        # convert ground truth to numpy array
        ground_truth = np.array(ground_truth).astype(np.int64)
        # convert the colors of the ground truth to the labels
        ground_truth = self.__convert_labels__(ground_truth)
       
        return image, ground_truth

    def __len__(self):
        return len(self.images)

    # define a method that returns the labels map, mapping each color to a label
    def __get_label_map__(self):
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
