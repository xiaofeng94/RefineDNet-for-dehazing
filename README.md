# RefineDNet for dehazing

RefineDNet is a two-stage dehazing framework which can be weakly supervised using real-world unpaired images. 
That is, the training set never requires paired hazy and haze-free images coming from the same scene.

In the first stage, it adopts DCP to restore visibility of the input hazy image. 
In the second stage, it improves the realness of preliminary results from the first stage via CNNs. 
RefineDNet is outlined in the following figure, and more details can be found in the [paper]() (coming soon)
![framework](https://github.com/xiaofeng94/RefineDNet_for_dehazing/blob/master/datasets/figures/framework_github.jpg)

# Our Environment
- Ubuntu 16.06
- Python (>= 3.5)
- PyTorch (>= 1.1.0) with CUDA 9.0
- torchvision (>=0.3.0)
- numpy (>= 1.17.0)

# Testing
## Download the pretrained models.
1. Get the model trained on RESIDE-unpaired [here]() (coming soon).

2. Create a folder named `checkpoints`, and unzip the pretained model in `./checkpoints`.
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
## Quick test on real-world images
1. Download the pretrained model on BeDDE.

2. Run the following command from <RefineDNet_root>.
```
python quick_test.py --dataroot ./datasets/quick_test --dataset_mode single --name refined_DCP_outdoor --model refined_DCP --phase test --preprocess none --save_image --method_name refined_DCP_outdoor_ep_60 --epoch 60
```
The results will be saved in the folder `<RefineDNet_root>/datatsets/quick_test/refined_DCP_outdoor_ep_60`.

## Test on BeDDE
1. Download the pretrained model on BeDDE.

2. Run the following command from `<RefineDNet_root>`.
```
python test_BeDDE.py --dataroot <BeDDE_root> --dataset_mode simple_bedde --bedde_list ./datasets/BeDDE/bedde_list.txt --name refined_DCP_outdoor --model refined_DCP --phase test --preprocess none --save_image --method_name refined_DCP_outdoor_ep_60 --epoch 60
```
The results will be saved in the folder `<BeDDE_root>/<city_name>/refined_DCP_outdoor_ep_60`.

# Training
## Train RefineDNet on RESIDE-unpaired
1. Download RESIDE-unpaired [here]() (comming soon), and unzip it in the folder <RefineDNet_root>/datasets.
Your directory tree should look like
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

3. Run the following command from `<RefineDNet_root>`.
```
python train.py --dataroot ./datasets/RESIDE-unpaired --dataset_mode unpaired --model refined_DCP --name refined_DCP_outdoor --niter 30 --niter_decay 60 --lr_decay_iters 10 --preprocess scale_min_and_crop --load_size 300 --crop_size 256 --num_threads 8 --save_epoch_freq 3 --lambda_G 0.05 --lambda_identity 1
```
## Train RefineDNet on ITS (from RESIDE-standard)
1. Download ITS [here](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0). Unzip hazy.zip and clear.zip into `<RefineDNet_root>/datasets/ITS`. 

2. Rename the hazy image folder as `trainA` and the clear image folder as `trainB`.
Then, your directory tree should look like
```
<RefineDNet_root>
├── datasets
│   ├── BeDDE
│   ├── ITS
│   │   ├── trainA
│   │   └── trainB
│   ...
...
```
3. Open visdom by `python -m visdom.server`

4. Run the following command from `<RefineDNet_root>`.
```
python train.py --dataroot ./datasets/ITS --dataset_mode unpaired --model refined_DCP --name refined_DCP_indoor --niter 30 --niter_decay 60 --lr_decay_iters 5 --preprocess scale_width_and_crop --load_size 372 --crop_size 256 --num_threads 8 --save_epoch_freq 1 --lambda_G 0.05 --lambda_identity 1
```

# Performance
Some dehazing samples from BeDDE and the Internet produced by various methods.
![dehazing samples](https://github.com/xiaofeng94/RefineDNet_for_dehazing/blob/master/datasets/figures/outdoor_com_github.jpg)
# Useful links
1. [RESIDE dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0)

2. [BeDDE dataset](https://github.com/xiaofeng94/BeDDE-for-defogging)

3. This code is based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
