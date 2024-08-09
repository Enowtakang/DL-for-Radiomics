import os
import numpy as np
import cv2


"""
Define paths to all training dataset folders
"""
root_path_to_datasets = "C:/Users/HP/PycharmProjects/MPH/1data/randomColorSpace/"
d1_normal = root_path_to_datasets + "RCS-d(1)/normal/"
d1_pneumonia = root_path_to_datasets + "RCS-d(1)/pneumonia/"
d2_normal = root_path_to_datasets + "RCS-d(2)/normal/"
d2_pneumonia = root_path_to_datasets + "RCS-d(2)/pneumonia/"
d3_normal = root_path_to_datasets + "RCS-d(3)/normal/"
d3_pneumonia = root_path_to_datasets + "RCS-d(3)/pneumonia/"
d4_normal = root_path_to_datasets + "RCS-d(4)/normal/"
d4_pneumonia = root_path_to_datasets + "RCS-d(4)/pneumonia/"


"""
Define a function to 
    - load and prepare images from a specific folder
"""


def load_image_dataset(path_to_folder, image_size=(369, 369)):
    image_filenames = os.listdir(path_to_folder)

    # initialize an empty list to store the images
    images = list()

    # load each image in the folder
    for filename in image_filenames:
        image_path = os.path.join(path_to_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        # normalize pixel values to [0, 1]
        # image = image.astype('float32') / 255.0

        images.append(np.array(image))

    original_images = np.stack(images)

    return original_images


"""
Define a function to perform 
    - RCS transformation on the brightest pixels 
    - of each loaded image
"""


def rcs_transformation(original_images):
    transformed_images = list()

    for image in original_images:
        transformed_image = np.copy(image)
        # random number between 20 and 50
        random_number = np.random.randint(20, 50)
        print(random_number)
        # transform image
        transformed_image += random_number

        transformed_images.append(transformed_image)

    transformed_images = np.array(transformed_images)

    return transformed_images


"""
Define a function to 
    - load images, 
    - perform gaussian noise transformation on each image
    - save all images
"""


def complete_process(path_to_folder):
    # load dataset
    original_images = load_image_dataset(path_to_folder)

    # transform images
    transformed_images = [
        rcs_transformation(image) for image in original_images]

    # save transformed images
    # save the new images to the appropriate folder
    for i, image in enumerate(transformed_images):
        filename = f"rcs_transformed_image_{i}.JPG"
        full_path = path_to_folder + filename
        cv2.imwrite(full_path, image)


complete_process(d4_pneumonia)
