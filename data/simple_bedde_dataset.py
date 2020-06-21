### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os, ntpath

import numpy as np
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset,

class SimpleBeDDEDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--bedde_list', required=True, type=str, help='image list of BeDDE')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data_list_file = opt.bedde_list

        listFile = open(self.data_list_file, 'r')
        self.imagePaths = listFile.read().split()
        listFile.close()

        self.I_size = len(self.imagePaths) 

        self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))
        self.toTensor = transforms.ToTensor()
      
    def __getitem__(self, index):        
        ### input A (label maps)
        # print('bedde id %d'%index)
        I_path = self.imagePaths[index]              
        I_img = Image.open(I_path).convert('RGB')
        params = get_params(self.opt, I_img.size)

        I_name = os.path.splitext(ntpath.basename(I_path))[0]
        cityName = I_name.split('_')[0]

        I_dir = ntpath.dirname(I_path)
        base_dir = ntpath.dirname(I_dir)

        J_path = os.path.join(base_dir, 'gt', '%s_clear.png'%cityName)
        J_img = Image.open(J_path).convert('RGB')

        base_dir = ntpath.dirname(I_dir)
        mask_path = os.path.join(base_dir, 'mask', '%s_mask.mat'%I_name)
        mask_info = sio.loadmat(mask_path)

        J_root = ntpath.dirname(ntpath.dirname(I_path))

        # apply image transformation
        real_I = self.transform(I_img)
        real_J = (self.toTensor(J_img) - 0.5) / 0.5
        
        return {'haze': real_I, 'clear': real_J, 'mask': mask_info['mask'], 
                'city': cityName, 'paths': I_path}
        # return {'haze': real_I , 'city': cityName, 'paths': curPath}

    def __len__(self):
        return self.I_size
