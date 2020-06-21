"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import os

import cv2

import torch.nn.functional as F

def synthesize_fog(J, t, A=None):
    """
    Synthesize hazy image base on optical model
    I = J * t + A * (1 - t)
    """

    if A is None:
        A = 1

    return J * t + A * (1 - t)

def reverse_fog(I, t, A=1, t0=0.01):
    """
    Recover haze-free image using hazy image and depth
    J = (I - A) / max(t, t0) + A
    """

    t_clamp = torch.clamp(t, t0, 1)
    J = (I-A) / t_clamp  + A
    return torch.clamp(J, -1, 1)


def fuse_images(real_I, rec_J, refine_J):
    """
    real_I, rec_J, and refine_J: Images with shape hxwx3
    """
    # realness features
    mat_RGB2YMN = np.array([[0.299,0.587,0.114],
                            [0.30,0.04,-0.35],
                            [0.34,-0.6,0.17]])

    recH,recW,recChl = rec_J.shape
    rec_J_flat = rec_J.reshape([recH*recW,recChl])
    rec_J_flat_YMN = (mat_RGB2YMN.dot(rec_J_flat.T)).T
    rec_J_YMN = rec_J_flat_YMN.reshape(rec_J.shape)

    refine_J_flat = refine_J.reshape([recH*recW,recChl])
    refine_J_flat_YMN = (mat_RGB2YMN.dot(refine_J_flat.T)).T
    refine_J_YMN = refine_J_flat_YMN.reshape(refine_J.shape)

    real_I_flat = real_I.reshape([recH*recW,recChl])
    real_I_flat_YMN = (mat_RGB2YMN.dot(real_I_flat.T)).T
    real_I_YMN = real_I_flat_YMN.reshape(real_I.shape)

    # gradient features
    rec_Gx = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    rec_Gy = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    rec_GM = np.sqrt(rec_Gx**2 + rec_Gy**2)

    refine_Gx = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    refine_Gy = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    refine_GM = np.sqrt(refine_Gx**2 + refine_Gy**2)

    real_Gx = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    real_Gy = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    real_GM = np.sqrt(real_Gx**2 + real_Gy**2)

    # similarity
    rec_S_V = (2*real_GM*rec_GM+160)/(real_GM**2+rec_GM**2+160)
    rec_S_M = (2*rec_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(rec_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    rec_S_N = (2*rec_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(rec_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    rec_S_R = (rec_S_M*rec_S_N).reshape([recH,recW])

    refine_S_V = (2*real_GM*refine_GM+160)/(real_GM**2+refine_GM**2+160)
    refine_S_M = (2*refine_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(refine_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    refine_S_N = (2*refine_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(refine_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    refine_S_R = (refine_S_M*refine_S_N).reshape([recH,recW])


    rec_S = rec_S_R*np.power(rec_S_V, 0.4)
    refine_S = refine_S_R*np.power(refine_S_V, 0.4)


    fuseWeight = np.exp(rec_S)/(np.exp(rec_S)+np.exp(refine_S))
    fuseWeightMap = fuseWeight.reshape([recH,recW,1]).repeat(3,axis=2)

    fuse_J = rec_J*fuseWeightMap + refine_J*(1-fuseWeightMap)
    return fuse_J



def get_tensor_dark_channel(img, neighborhood_size):
    shape = img.shape
    if len(shape) == 4:
        img_min = torch.min(img, dim=1)
        img_dark = F.max_pool2d(img_min, kernel_size=neighborhood_size, stride=1)
    else:
        raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

    return img_dark



def array2Tensor(in_array, gpu_id=-1):
    in_shape = in_array.shape
    if len(in_shape) == 2:
        in_array = in_array[:,:,np.newaxis]

    arr_tmp = in_array.transpose([2,0,1])
    arr_tmp = arr_tmp[np.newaxis,:]

    if gpu_id >= 0:
        return torch.tensor(arr_tmp.astype(np.float)).to(gpu_id)
    else:
        return torch.tensor(arr_tmp.astype(np.float))


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def rescale_tensor(input_tensor):
    """"Converts a Tensor array into the Tensor array whose data are identical to the image's.
    [height, width] not [width, height]

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tmp = input_tensor.cpu().float()
        output_tmp = (input_tmp + 1) / 2.0 * 255.0
        output_tmp = output_tmp.to(torch.uint8)
    else:
        return input_tensor

    return output_tmp.to(torch.float32) / 255.0

    # if not isinstance(input_image, np.ndarray):
    #     if isinstance(input_image, torch.Tensor):  # get the data from a variable
    #         image_tensor = input_image.data
    #     else:
    #         return input_image
    #     image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    #     image_numpy = (image_numpy + 1) / 2.0 * white_color  # post-processing: tranpose and scaling
    # else:  # if it is a numpy array, do nothing
    #     image_numpy = input_image
    # return torch.from_numpy(image_numpy)

def my_imresize(in_array, tar_size):
    oh = in_array.shape[0]
    ow = in_array.shape[1]

    if len(tar_size) == 2:
        h_ratio = tar_size[0]/oh
        w_ratio = tar_size[1]/ow
    elif len(tar_size) == 1:
        h_ratio = tar_size
        w_ratio = tar_size

    if len(in_array.shape) == 3:
        return ndimage.zoom(in_array, (h_ratio, w_ratio, 1), prefilter=False)
    else:
        return ndimage.zoom(in_array, (h_ratio, w_ratio), prefilter=False)

def psnr(img, ref, max_val=1):
    if isinstance(img, torch.Tensor):
        distImg = img.cpu().float().numpy()
    elif isinstance(img, np.ndarray):
        distImg = img.astype(np.float)
    else:
        distImg = np.array(img).astype(np.float)

    if isinstance(ref, torch.Tensor):
        refImg = ref.cpu().float().numpy()
    elif isinstance(ref, np.ndarray):
        refImg = ref.astype(np.float)
    else:
        refImg = np.array(ref).astype(np.float)

    rmse = np.sqrt( ((distImg-refImg)**2).mean() )
    # rmse = np.std(distImg-refImg) # keep the same with RESIDE's criterion
    return 20*np.log10(max_val/rmse)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
