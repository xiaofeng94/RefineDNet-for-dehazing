import os, ntpath

import numpy as np
import scipy.io as sio
import torchvision.utils as vutils

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.nThreads = 1   # mytest code only supports nThreads = 1
    opt.batchSize = 1  # mytest code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        real_I = util.tensor2im(data['haze'], np.float) # [0, 255], np.float
        real_J = util.tensor2im(data['clear'], np.float) # [0, 255], np.float

        rec_J = util.tensor2im(visuals['rec_J'], np.float) # [0, 255], np.float
        refine_J = util.tensor2im(visuals['refine_J'], np.float) # [0, 255], np.float

        result_J = util.fuse_images(real_I, rec_J, refine_J) # [0, 255], np.float

        img_paths = model.get_image_paths()     # get image paths
        short_path = ntpath.basename(img_paths[0])
        name = os.path.splitext(short_path)[0]

        print('processing image %s (%d/%d)'%(short_path, i+1, len(dataset)))

        if opt.save_image:
            curSaveFolder = os.path.join(opt.dataroot, data['city'][0], opt.method_name)
            if not os.path.exists(curSaveFolder):
                os.makedirs(curSaveFolder, mode=0o777)

            dehzImg = (result_J).astype(np.uint8) #[0, 255], np.uint8
            util.save_image(dehzImg, os.path.join(curSaveFolder, '%s_%s.png'%(name, opt.method_name)))

