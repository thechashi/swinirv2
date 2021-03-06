import os
import scipy.misc as misc
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
def transform(image, resize_size):
        resize_image = cv2.resize(image, resize_size, interpolation=cv2.INTER_CUBIC)
        return np.array(resize_image) 

def get_npz(file_name):
    image = np.load(file_name)
    image = image.f.arr_0
    return image
def view_npz(file_name):
    image = np.load(file_name)
    image = image.f.arr_0
    plt.ion()
    plt.figure()
    plt.imshow(image)
def summary_2D_npz(file_name):
    image = np.load(file_name)
    image = image.f.arr_0
    shape, dtype, min_, max_, mean, std = image.shape, image.dtype, image.min(), image.max(), np.mean(image), np.std(image)
    print('Img shape: ',shape)    # shape of an image
    print('Dtype: ', dtype)
    print('Min: ', min_)
    print('Max: ', max_)
    print('Std: ', std)
    print('Mean: ', mean)    
def npz_to_png(input_path, output_path):
    image = np.load(input_path)
    image = image.f.arr_0
    cv2.imwrite(output_path, image.reshape((image.shape[0], image.shape[1], 1)))
def resize_all(input_path, output_path):
    folders = ['/HR', '/LR_bicubic/X2', '/LR_bicubic/X3', '/LR_bicubic/X4', '/LR_bicubic/X8']
    for folder in folders:
        if not os.path.exists(output_path + folder):
            os.makedirs(output_path + folder)
    print('Processing HR images...')
    # making HR images
    for filename in os.listdir(input_path):
        if '.npz' in filename:
            file_path = os.path.join(input_path, filename)
            img = get_npz(file_path)
            shape = img.shape
            new_shape = ((shape[1]//24)*24, (shape[0]//24)*24)
            img = transform(img, new_shape)
            np.savez(os.path.join(output_path + folders[0], filename), img)
    sizes = [2, 3, 4, 8]
    # making LR 2, 3, 4, 8 images
    print('Processing LR images...')
    for filename in os.listdir(output_path + folders[0]):
        file_path = os.path.join(output_path + folders[0], filename)
        img = get_npz(file_path)
        shape = img.shape
        idx = 1
        for size in sizes:
            new_shape = ((shape[1]//size), (shape[0]//size))
            img = transform(img, new_shape)
            np.savez(os.path.join(output_path + folders[idx], filename.split('.')[0]+'x{}.npz'.format(size)), img)
            idx += 1
        
if __name__ == "__main__":
    resize_all('../testsets/act',  '../testsets/act_2')
# =============================================================================
#     filename = '../testsets/code_test/LR_bicubic/X8/0x8.npz'
#     view_npz(filename)
#     summary_2D_npz(filename)
# =============================================================================
# =============================================================================
#     summary_2D_npz('../testsets/slices/LR_bicubic/X3/0x3.npz')
#     npz_to_png('../testsets/slices/LR_bicubic/X2/0x2.npz', '../testsets/slices/LR_bicubic/X2/0x2.png')
#     npz_to_png('../testsets/slices/HR/0.npz', '../testsets/slices/HR/0.png')
# =============================================================================
# =============================================================================
#     for i in range(9):
#         img = get_npz('../testsets/slices/HR/{}.npz'.format(i))
#         print(img.shape) 
# # =============================================================================
# #         plt.ion()
# #         plt.figure()
# #         plt.imshow(img)
# # =============================================================================
#         img = transform(img, (152,312))
#         np.savez('../testsets/slices/LR_bicubic/X3/{}x3.npz'.format(i), img)
# # =============================================================================
# #         plt.ion()
# #         plt.figure()
# #         plt.imshow(img)
# # =============================================================================
#         print(img.shape)
# =============================================================================
# -*- coding: utf-8 -*-

