import cv2
import glob
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


"""
1. Paths to raw training images
"""
datasets_path = "C:/Users/HP/PycharmProjects/MPH/datasets/"
d1_normal = datasets_path + "d(1)/normal/*.*"
d1_pneumonia = datasets_path + "d(1)/pneumonia/*.*"
d2_normal = datasets_path + "d(2)/normal/*.*"
d2_pneumonia = datasets_path + "d(2)/pneumonia/*.*"
d3_normal = datasets_path + "d(3)/normal/*.*"
d3_pneumonia = datasets_path + "d(3)/pneumonia/*.*"
d4_normal = datasets_path + "d(4)/normal/*.*"
d4_pneumonia = datasets_path + "d(4)/pneumonia/*.*"

"""
2. Paths to PCA compressed training images
"""
compressed_path = "C:/Users/HP/PycharmProjects/MPH/1data/"
d1_pca_normal = compressed_path + "d(1)/normal/"
d1_pca_pneumonia = compressed_path + "d(1)/pneumonia/"
d2_pca_normal = compressed_path + "d(2)/normal/"
d2_pca_pneumonia = compressed_path + "d(2)/pneumonia/"
d3_pca_normal = compressed_path + "d(3)/normal/"
d3_pca_pneumonia = compressed_path + "d(3)/pneumonia/"
d4_pca_normal = compressed_path + "d(4)/normal/"
d4_pca_pneumonia = compressed_path + "d(4)/pneumonia/"
normal = "normal"
pneumonia = "pneumonia"


"""
3. Compression function
"""


def compress_img_with_pca(
        source_path, destination_path, im_class):
    noc = 400  # number of components
    count = 0
    size = 128

    for file in glob.glob(source_path):
        """Read image and convert to greyscale"""
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image.shape)
        """obtain image dimensions"""
        height, width = image.shape
        """fit and transform image """
        pca = PCA(noc).fit(image)
        image_transformed = pca.transform(image)
        # print(image_transformed.shape)
        """inverse transform image"""
        image_inverse_transformed = pca.inverse_transform(
            image_transformed)
        """reconstruct image"""
        image_reconstructing = np.reshape(
            image_inverse_transformed,
            (height, width))

        """resize image """
        image2 = cv2.resize(
            image_reconstructing, (size, size))

        count += 1

        plt.axis('off')
        plt.imshow(image2.astype('uint8'))

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        plt.savefig(
            f"{destination_path}_{im_class}_{count}.JPG",
            bbox_inches='tight',
            pad_inches=0.0)


compress_img_with_pca(
    d4_pneumonia, d4_pca_pneumonia, pneumonia)

"""
REFERENCES:
    - How to batch process multiple images in python:
        https://www.youtube.com/watch?v=QxzxLVzNfbI
    - PCA image compression with python:
        https://www.youtube.com/watch?v=L2oyN8-j6s4
"""
