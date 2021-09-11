from math import log10, sqrt
import cv2
import numpy as np
import os

from statistics import mean

l1 = []
l2 = []


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def driver(path1, path2):
    # replace with the folder containing images and iterate over it.
    original = cv2.imread("./images/" + path1)
    # replace with the folder containing bad images and iterate over it.
    compressed = cv2.imread("./masks/" + path2, 1)
    new_compressed = cv2.resize(compressed, (0, 0), fx=0.5, fy=0.5)
    # to be fed into list and average calculated from it.
    psnr_value = PSNR(original, new_compressed)
    ssim_value = calculate_ssim(original, new_compressed)
    print(f"PSNR value is {psnr_value} dB")
    print(f"SSIM value is : {ssim_value}\n")
    l1.append(psnr_value)
    l2.append(ssim_value)


def main():

    path = "./"
    # !ls
    path1 = "images"  # Original images
    path2 = "masks"  # Target/Result images
    merged_list = tuple(zip(os.listdir(path + path1),
                        os.listdir(path + path2)))
    for i in merged_list:
        driver(i[0], i[1])

    print(merged_list)
    print("Average PSNR value : ", round(mean(l1), 4))
    print("Average SSIM value : ", round(mean(l2), 4))


if__name__ == "__main__":
    main()
