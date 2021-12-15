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

if __name__ == "__main__":
    for i in range(9):
        img = get_npz('../testsets/slices/HR/{}.npz'.format(i))
        print(img.shape) 
# =============================================================================
#         plt.ion()
#         plt.figure()
#         plt.imshow(img)
# =============================================================================
        img = transform(img, (152,312))
        np.savez('../testsets/slices/LR_bicubic/X3/{}x3.npz'.format(i), img)
# =============================================================================
#         plt.ion()
#         plt.figure()
#         plt.imshow(img)
# =============================================================================
        print(img.shape)
