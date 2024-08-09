from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from PIL import Image
import numpy as np
import os


"""
Define the geometric transformation rules
"""
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect',
    # also try: nearest, constant, wrap
    cval=0,)


"""
Define the source directories for the images
"""
root = "C:/Users/HP/PycharmProjects/MPH/1data/3G/"
g_d1_normal = root + "G-d(1)/normal/"
g_d1_pneumonia = root + "G-d(1)/pneumonia/"
g_d2_normal = root + "G-d(2)/normal/"
g_d2_pneumonia = root + "G-d(2)/pneumonia/"
g_d3_normal = root + "G-d(3)/normal/"
g_d3_pneumonia = root + "G-d(3)/pneumonia/"
g_d4_normal = root + "G-d(4)/normal/"
g_d4_pneumonia = root + "G-d(4)/pneumonia/"


"""
Create a dataset batch to be used 
    for each image directory
"""
dataset = list()
size = 128


def make_dataset(directory):
    images = os.listdir(directory)

    for i, image_name in enumerate(images):
        if image_name.split('.')[1] == 'JPG':
            image = io.imread(directory + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((size, size))
            dataset.append(np.array(image))


"""
Augment images in the directory
"""
# note that batch_size
# (number of images to be included/used) = 51, and
# number of transformations per item/image in batch = 1
batch_size = 51
save_prefix = 'augmented'
save_format = 'jpg'


def augment(np_array, directory):
    i = 0

    for batch in datagen.flow(
        np_array,
        batch_size=batch_size,
        save_to_dir=directory,
        save_prefix=save_prefix,
        save_format=save_format,
    ):
        i += 1
        if i == 1:
            break


"""
Make results
"""


def make_results(directory):
    make_dataset(directory)
    x = np.array(dataset)
    augment(x, directory)


# make_results(g_d4_pneumonia)


"""
REFERENCES:
    - https://www.youtube.com/watch?v=ccdssX4rIh8
"""
