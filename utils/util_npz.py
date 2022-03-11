import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
def transform(image, resize_size):
    """
    Bicubic resize

    Parameters
    ----------
    image : np.array
        image to be resized.
    resize_size : tuple
        new size.

    Returns
    -------
    np.array
        resized image.

    """
    resize_image = cv2.resize(image, resize_size, interpolation=cv2.INTER_CUBIC)
    return np.array(resize_image) 

def get_npz(file_name):
    """
    Loads an image from a npz file

    Parameters
    ----------
    file_name : str
        file to be loaded.

    Returns
    -------
    image : np.array
        image from npz file.

    """
    image = np.load(file_name)
    image = image.f.arr_0
    return image

def view_npz(file_name):
    """
    Visualize an npz file

    Parameters
    ----------
    file_name : str
        file path.

    Returns
    -------
    None.

    """
    image = np.load(file_name)
    image = image.f.arr_0
    plt.ion()
    plt.figure()
    plt.imshow(image)
    
def summary_2D_npz(file_name):
    """
    Summarize an npz file

    Parameters
    ----------
    file_name : str
        file path.

    Returns
    -------
    None.

    """
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
    """
    Npz to png conversion

    Parameters
    ----------
    input_path : str
        input path.
    output_path : str
        ouptut path.

    Returns
    -------
    None.

    """
    image = np.load(input_path)
    image = image.f.arr_0
    cv2.imwrite(output_path, image.reshape((image.shape[0], image.shape[1], 1)))
    
def all_npz_to_png(input_path, output_path):
    for filename in os.listdir(input_path):
        if '.npz' in filename:
            file_path = os.path.join(input_path, filename)
            output_img_path = os.path.join(output_path, filename.split('.')[0] + '.png')
            image = np.load(file_path)
            image = image.f.arr_0
            cv2.imwrite(output_img_path, image.reshape((image.shape[0], image.shape[1], 1)))
            
def resize_all(input_path, output_path):
    """
    Resize all npz files into HR, /2, /3, /4, and /8 sizes 

    Parameters
    ----------
    input_path : str
        input directory.
    output_path : str
        output directory.

    Returns
    -------
    None.

    """
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
            
def get_abs_max(input_path, verbose=False):
    """
    Get absolute max pixel value of a npz dataset

    Parameters
    ----------
    input_path : str
        input directory.
    verbose : bool, optional
        Prints fileby file details. The default is False.

    Returns
    -------
    dataset_max : float
        abs max pixel value of the dataset.

    """
    dataset_max = -math.inf
    for filename in os.listdir(input_path):
        if '.npz' in filename:
            file_path = os.path.join(input_path, filename)
            image = get_npz(file_path)
            min_ = image.min()
            max_ = image.max()
            current_max = max(abs(max_), abs(min_))
            if verbose:
                print('Filename: {}, min: {}, max: {}, absolute max: {}'.format(filename, min_, max_, current_max))
            dataset_max = max(dataset_max, current_max)
    if verbose:
        print('Datset max: ', dataset_max)
    return dataset_max
        
        
if __name__ == "__main__":
    #resize_all('../testsets/slices/raw',  '../testsets/slices')
    #print(get_abs_max('../trainsets/earth1_ds/HR'))
    all_npz_to_png('../testsets/slices/LR_bicubic/X4', '../testsets/slices/LR_bicubic/X4')
    pass
