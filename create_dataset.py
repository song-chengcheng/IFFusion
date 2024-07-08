from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import re
from utils import randrot, randfilp
from natsort import natsorted
# from model.dark_isp import Low_Illumination_Degrading

class MSRSData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, dataroot, is_train=True, crop=lambda x: x):
        super(MSRSData, self).__init__()
        self.is_train = is_train
        if is_train:
            self.vis_folder = os.path.join(dataroot, 'train', 'vi')
        else:
            self.vis_folder = os.path.join(dataroot, 'test', 'vi')
        self.crop = torchvision.transforms.RandomCrop(256)  # 随机裁剪
        # gain infrared and visible images list
        self.vis_list = natsorted(os.listdir(self.vis_folder))
        print(len(self.vis_list))
        #self.ST = SpatialTransformer(self.crop.size[0],self.crop.size[0],False)

    def __getitem__(self, index):
        # gain image path
        image_name = self.vis_list[index]
        data_class = re.split('[.]', image_name)[0][-1]
        if data_class == 'D':
            class_data = 1
        else:
            class_data = 0
        vis_path = os.path.join(self.vis_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)

        if self.is_train:
            ## 训练图像进行一定的数据增强，包括翻转，旋转，以及随机裁剪等
            vis_ir = vis
            if vis_ir.shape[-1]<=256 or vis_ir.shape[-2]<=256:
                vis_ir=TF.resize(vis_ir,256)
            vis_ir = randfilp(vis_ir)
            vis_ir = randrot(vis_ir)
            patch = self.crop(vis_ir)
            vis = patch
            return vis.squeeze(0), class_data
        else: 

            return vis.squeeze(0), class_data

    def __len__(self):
        return len(self.vis_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
    

class Fusion_Data(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, dataroot, is_train=True, crop=lambda x: x):
        super(Fusion_Data, self).__init__()
        self.is_train = is_train
        if is_train:
            self.vis_folder = os.path.join(dataroot, 'train', 'vi')
            self.ir_folder = os.path.join(dataroot, 'train', 'ir')
            self.low_folder = os.path.join(dataroot, 'train', 'test')
            # self.en_folder = os.path.join(dataroot, 'train', 'en')
        else:
            self.vis_folder = os.path.join(dataroot, 'test', 'vi')
            self.ir_folder = os.path.join(dataroot, 'test', 'ir')
        self.crop = torchvision.transforms.RandomCrop(256)  # 随机裁剪
        # gain infrared and visible images list
        self.vis_list = natsorted(os.listdir(self.vis_folder))
        print(len(self.vis_list))
        #self.ST = SpatialTransformer(self.crop.size[0],self.crop.size[0],False)

    def __getitem__(self, index):
        # gain image path
        image_name = self.vis_list[index]
        file_name = re.split('[.]', image_name)
        if self.is_train:
            low_path = os.path.join(self.low_folder, file_name[0])
            low_list = os.listdir(low_path)
            low_list = random.sample(low_list, len(low_list))
            vi1_path = os.path.join(low_path, low_list[0])
            vi2_path = os.path.join(low_path, low_list[1])
        # data_class = re.split('[.]', image_name)[0][-1]
        # if data_class == 'D':
        #     class_data = 1
        # else:
        #     class_data = 0
        # if self.is_train:
        #     en_path = os.path.join(self.en_folder, image_name)
        #     en = self.imread(path=en_path)
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)


        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        if self.is_train:
            vi1 = self.imread(path=vi1_path)
            # print(vi2_path)
            vi2 = self.imread(path=vi2_path)
            # dark_vi, _ = Low_Illumination_Degrading(vis.squeeze(0))
            # dark_vi = dark_vi.unsqueeze(0)
            # if random.randint(0, 1):
            #     vi1 = self.imread(path=vi1_path)
            # else:
            #     vi1 = vis
            # vi2 = self.imread(path=vi2_path)
        if self.is_train:
            ## 训练图像进行一定的数据增强，包括翻转，旋转，以及随机裁剪等
            vis_ir = torch.cat([vis, vi1, vi2, ir], dim=1)
            if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
                vis_ir = TF.resize(vis_ir, 256)
            vis_ir = randfilp(vis_ir)
            vis_ir = randrot(vis_ir)
            patch = self.crop(vis_ir)
            vis, vi1, vi2, ir = torch.split(patch, [3, 3, 3, 3], dim=1)
            return vis.squeeze(0), vi1.squeeze(0), vi2.squeeze(0), ir.squeeze(0)
        else:

            return vis.squeeze(0), ir.squeeze(0), image_name

    def __len__(self):
        return len(self.vis_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else: ## infrared images single channel
                img = Image.open(path).convert('L')
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts

class FusionData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """
    
    def __init__(self, opts, crop=lambda x: x):
        super(FusionData, self).__init__()          
        self.vis_folder = os.path.join(opts.dataroot, opts.dataname, 'test', 'vi')
        self.ir_folder = os.path.join(opts.dataroot, opts.dataname, 'test', 'ir')
        # self.vis_folder = os.path.join(opts.dataroot, opts.dataname, 'vi')
        # self.ir_folder = os.path.join(opts.dataroot, opts.dataname, 'ir')
        
        # self.vis_folder = os.path.join('/data/timer/Segmentation/SegNext/datasets/MSRS/RGB')
        # self.ir_folder = os.path.join('/data/timer/Segmentation/SegNext/datasets/MSRS/Thermal')
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)     
        # read image as type Tensor
        vis, w, h = self.imread(path=vis_path)
        ir, w, h = self.imread(path=ir_path, vis_flage=False)
        return ir.squeeze(0), vis.squeeze(0), image_name, w, h

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            # 获取图像大小
            width, height = img.size

            # 调整图像大小到32的倍数
            new_width = width - (width % 32)
            new_height = height - (height % 32)
            img = img.resize((new_width, new_height))
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                # 获取图像大小
                width, height = img.size

                # 调整图像大小到32的倍数
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                # 获取图像大小
                width, height = img.size

                # 调整图像大小到32的倍数
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts, width, height
