# Under construction
# RefineDNet_for_dehazing

RefineDNet is a two-stage dehazing framework which can be weakly supervised using real-world unpaired images. 
That is, the training set never requires paired hazy and haze-free images coming from the same scene.

In the first stage, it adopts DCP to restore visibility of the input hazy image. 
In the second stage, it improves the realness of preliminary results from the first stage via CNNs.

# Our Environment
- Ubuntu 16.06
- Python (>= 3.5)
- pytorch (>= 1.1.0) with CUDA 9.0
- torchvision (>=0.3.0)
- numpy
- scipy
- skimage

# Testing
## Test on BeDDE
1. Download BeDDE [here](https://github.com/xiaofeng94/BeDDE-for-defogging). 

2. Download the pretrained model [here]() (coming soon).

3. Create a folder named `checkpoints`, and unzip the pretained model in `./checkpoints`.
Now, your directory tree should look like
```
<RefineDNet_root>
├── checkpoints
│   ├── refined_DCP_outdoor
│   │   ├── 60_net_D.pth
│   │   ├── 60_net_Refiner_J.pth
│   │   ├── 60_net_Refiner_T.pth
│   │   └── test_opt.txt
│   ...
...
```
4. Run the following command from <RefineDNet_root>.
```
python test_BeDDE.py --dataroot <BeDDE_root> --dataset_mode simple_bedde --bedde_list ./datasets/BeDDE/bedde_list.txt --name refined_DCP_outdoor --model refined_DCP --phase test --preprocess none --save_image --method_name refined_DCP_outdoor_ep_60 --epoch 60
```
The results will be saved in `<BeDDE_root>/<city_name>/refined_DCP_outdoor_ep_60`.

# Training
## Train RefineDNet on RESIDE-unpaired
1. Download RESIDE-unpaired [here]() (comming soon), and unzip it in the folder <RefineDNet_root>/datasets.
your directory tree should look like
```
<RefineDNet_root>
├── datasets
│   ├── BeDDE
│   ├── RESIDE-unpaired
│   │   ├── trainA
│   │   └── trainB
│   ...
...
```
2. Open visdom by `python -m visdom.server`

3. Run the following command from <RefineDNet_root>.
```
python train.py --dataroot ./datasets/RESIDE-unpaired --dataset_mode unpaired --model refined_DCP --netR_T unet_trans_256 --netR_J resnet_9blocks --name refined_DCP_3 --niter 30 --niter_decay 60 --lr_decay_iters 10 --preprocess scale_min_and_crop --load_size 300 --crop_size 256 --num_threads 8 --save_epoch_freq 3 --lambda_G 0.05 --lambda_identity 1
```

# References
TODO
