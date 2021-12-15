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
 
if __name__ == "__main__":
    view_npz('../testsets/slices/LR_bicubic/X3/0x3.npz')
    summary_2D_npz('../testsets/slices/LR_bicubic/X3/0x3.npz')
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
