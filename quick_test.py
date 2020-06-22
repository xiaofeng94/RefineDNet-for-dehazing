import os,time
import ntpath

import numpy as np
import scipy.io as sio

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    if opt.save_image:
        curSaveFolder = os.path.join(opt.dataroot, opt.method_name)
        if not os.path.exists(curSaveFolder):
            os.makedirs(curSaveFolder, mode=0o777)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    time_total = 0
    for i, data in enumerate(dataset):
        # if i <= 627:
        #     continue

        img_path = data['paths']
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        print('%s [%d]'%(short_path, i+1))
        # print(data['B_paths'])

        if 'haze' in data.keys():
            minSize = min(data['haze'].shape[2:4])
        else:
            minSize = min(data['A'].shape[2:4])
        if minSize < 256:
            print(' skip because the minimum size is %s'%minSize)
            continue

        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        t0 = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        time_total += time.time() - t0

        visuals = model.get_current_visuals()  # get image results

        rec_J = util.tensor2im(visuals['rec_J'], np.float)/255. # [0, 1]
        refine_J = util.tensor2im(visuals['refine_J'], np.float)/255. # [0, 1]
        real_I = util.tensor2im(data['haze'], np.float) # [0, 255], np.float
        result_J = util.fuse_images(real_I, rec_J*255., refine_J*255.)/255. # [0, 1], np.float

        # save result images
        if opt.save_image:
            dehzImg = (result_J*255).astype(np.uint8) #[0, 255], np.uint8
            util.save_image(dehzImg, os.path.join(curSaveFolder, '%s_dehz.png'%(name)))

            # refinedT = util.tensor2im(visuals['refine_T_vis'])
            # util.save_image(refinedT, os.path.join(curSaveFolder, '%s_ref_T.png'%(name)))

    print('num: %d'%len(dataset))
    print('average time: %f'%(time_total/len(dataset)))
