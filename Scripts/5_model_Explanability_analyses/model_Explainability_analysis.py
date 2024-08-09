import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tensorflow.keras.models import load_model
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


"""
Paths to models
"""
# paths to all 32 models
root_path_to_models = "C:/Users/HP/PycharmProjects/MPH/2models/"
none = root_path_to_models + "_none/"
geometric = root_path_to_models + "geometric/"
noise_gaussian = root_path_to_models + "noise_gaussian/"
random_color_space = root_path_to_models + "random_color_space/"

D_1_AlexNet = none + "D(1)_AlexNet.h5"
D_1_MobileNet = none + "D(1)_MobileNet.h5"
D_2_AlexNet = none + "D(2)_AlexNet.h5"
D_2_MobileNet = none + "D(2)_MobileNet.h5"
D_3_AlexNet = none + "D(3)_AlexNet.h5"
D_3_MobileNet = none + "D(3)_MobileNet.h5"
D_4_AlexNet = none + "D(4)_AlexNet.h5"
D_4_MobileNet = none + "D(4)_MobileNet.h5"

G_D_1_AlexNet = geometric + "G-D(1)_AlexNet.h5"
G_D_1_MobileNet = geometric + "G-D(1)_MobileNet.h5"
G_D_2_AlexNet = geometric + "G-D(2)_AlexNet.h5"
G_D_2_MobileNet = geometric + "G-D(2)_MobileNet.h5"
G_D_3_AlexNet = geometric + "G-D(3)_AlexNet.h5"
G_D_3_MobileNet = geometric + "G-D(3)_MobileNet.h5"
G_D_4_AlexNet = geometric + "G-D(4)_AlexNet.h5"
G_D_4_MobileNet = geometric + "G-D(4)_MobileNet.h5"

N_D_1_AlexNet = noise_gaussian + "GN-D(1)_AlexNet.h5"
N_D_1_MobileNet = noise_gaussian + "GN-D(1)_MobileNet.h5"
N_D_2_AlexNet = noise_gaussian + "GN-D(2)_AlexNet.h5"
N_D_2_MobileNet = noise_gaussian + "GN-D(2)_MobileNet.h5"
N_D_3_AlexNet = noise_gaussian + "GN-D(3)_AlexNet.h5"
N_D_3_MobileNet = noise_gaussian + "GN-D(3)_MobileNet.h5"
N_D_4_AlexNet = noise_gaussian + "GN-D(4)_AlexNet.h5"
N_D_4_MobileNet = noise_gaussian + "GN-D(4)_MobileNet.h5"

R_D_1_AlexNet = random_color_space + "RCS-D(1)_AlexNet.h5"
R_D_1_MobileNet = random_color_space + "RCS-D(1)_MobileNet.h5"
R_D_2_AlexNet = random_color_space + "RCS-D(2)_AlexNet.h5"
R_D_2_MobileNet = random_color_space + "RCS-D(2)_MobileNet.h5"
R_D_3_AlexNet = random_color_space + "RCS-D(3)_AlexNet.h5"
R_D_3_MobileNet = random_color_space + "RCS-D(3)_MobileNet.h5"
R_D_4_AlexNet = random_color_space + "RCS-D(4)_AlexNet.h5"
R_D_4_MobileNet = random_color_space + "RCS-D(4)_MobileNet.h5"

paths_to_models_dictionary = {
    'D(1)-AlexNet': D_1_AlexNet, 'D(1)-MobileNet': D_1_MobileNet,
    'D(2)-AlexNet': D_2_AlexNet, 'D(2)-MobileNet': D_2_MobileNet,
    'D(3)-AlexNet': D_3_AlexNet, 'D(3)-MobileNet': D_3_MobileNet,
    'D(4)-AlexNet': D_4_AlexNet, 'D(4)-MobileNet': D_4_MobileNet,

    'G-D(1)-AlexNet': G_D_1_AlexNet, 'G-D(1)-MobileNet': G_D_1_MobileNet,
    'G-D(2)-AlexNet': G_D_2_AlexNet, 'G-D(2)-MobileNet': G_D_2_MobileNet,
    'G-D(3)-AlexNet': G_D_3_AlexNet, 'G-D(3)-MobileNet': G_D_3_MobileNet,
    'G-D(4)-AlexNet': G_D_4_AlexNet, 'G-D(4)-MobileNet': G_D_4_MobileNet,

    'N-D(1)-AlexNet': N_D_1_AlexNet, 'N-D(1)-MobileNet': N_D_1_MobileNet,
    'N-D(2)-AlexNet': N_D_2_AlexNet, 'N-D(2)-MobileNet': N_D_2_MobileNet,
    'N-D(3)-AlexNet': N_D_3_AlexNet, 'N-D(3)-MobileNet': N_D_3_MobileNet,
    'N-D(4)-AlexNet': N_D_4_AlexNet, 'N-D(4)-MobileNet': N_D_4_MobileNet,

    'R-D(1)-AlexNet': R_D_1_AlexNet, 'R-D(1)-MobileNet': R_D_1_MobileNet,
    'R-D(2)-AlexNet': R_D_2_AlexNet, 'R-D(2)-MobileNet': R_D_2_MobileNet,
    'R-D(3)-AlexNet': R_D_3_AlexNet, 'R-D(3)-MobileNet': R_D_3_MobileNet,
    'R-D(4)-AlexNet': R_D_4_AlexNet, 'R-D(4)-MobileNet': R_D_4_MobileNet, }


"""
Paths to images
"""
normal_image_path = "C:/Users/HP/PycharmProjects/MPH/datasets/test/normal/IM-0001-0001.jpeg"
pneumonia_image_path = "C:/Users/HP/PycharmProjects/MPH/datasets/test/pneumonia/person1_virus_6.jpeg"

"""
Paths to results
"""
root_path_to_results = "C:/Users/HP/PycharmProjects/MPH/4results/"
cam_results_path = root_path_to_results + "4_class_activation_maps/"
osa_results_path = root_path_to_results + "5_occlusion_sensitivity_analyses/"

"""
Set the image size for each model
"""
image_size_for_each_model = {
    'AlexNet': (227, 227),
    'MobileNet': (224, 224), }

"""
Function to preprocess images
"""


def preprocess_image(image_path, target_size):
    print(f"Preprocessing image: {image_path} with target size: {target_size}")

    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)


"""
Function to perform occlusion sensitivity analysis
"""
patch_Size = 15
smooth_Samples = 20
smooth_Noise = 0.20


def occlusion_sensitivity_analysis(
        model, image, class_index, patch_size=patch_Size):

    saliency = Saliency(
        model, model_modifier=ReplaceToLinear(), clone=True)

    score = CategoricalScore([class_index])

    saliency_map = saliency(
        score, image, smooth_samples=smooth_Samples,
        smooth_noise=smooth_Noise)

    return saliency_map[0]


"""
Function to generate class activation maps
"""
penultimate_Layer = -1


def generate_cam(model, image, class_index):
    grad_cam = Gradcam(
        model, model_modifier=ReplaceToLinear(), clone=True)

    score = CategoricalScore([class_index])

    cam = grad_cam(score, image, penultimate_layer=penultimate_Layer)
    return cam[0]


"""
Load models and perform analyses
"""
results = list()


def evaluate_each_model():
    for model_name, path_to_model in paths_to_models_dictionary.items():
        """ 
        load the model
        """
        model = load_model(path_to_model)

        target_size = image_size_for_each_model['AlexNet'] if 'AlexNet' in model_name else image_size_for_each_model[
            'MobileNet']

        """
        Preprocess images
        """
        normal_image = preprocess_image(
            normal_image_path, target_size)
        pneumonia_image = preprocess_image(
            pneumonia_image_path, target_size)

        """
        Perform occlusion sensitivity analyses
        """
        normal_saliency = occlusion_sensitivity_analysis(
            model, normal_image, class_index=0)
        pneumonia_saliency = occlusion_sensitivity_analysis(
            model, pneumonia_image, class_index=1)

        """
        save saliency maps
        """
        # saliency map for normal image
        plt.imshow(normal_saliency, cmap='hot')
        plt.title(f'Saliency Map for {model_name} - Normal')
        # plt.colorbar()
        full_path_osa = os.path.join(
            osa_results_path,
            f'occlusion_sensitivity_{model_name}_normal.JPG')
        plt.savefig(full_path_osa)

        # saliency map for pneumonia image
        plt.imshow(pneumonia_saliency, cmap='jet')
        plt.title(f'Saliency Map for {model_name} - Pneumonia')
        # plt.colorbar()
        full_path_osa2 = os.path.join(
            osa_results_path,
            f'occlusion_sensitivity_{model_name}_pneumonia.JPG')
        plt.savefig(full_path_osa2)

        """
        Generate Class Activation Maps
        """
        normal_cam = generate_cam(
            model, normal_image, class_index=0)
        pneumonia_cam = generate_cam(
            model, pneumonia_image, class_index=1)

        """
        Save CAMs
        """
        # CAM for normal image
        plt.imshow(normal_cam, cmap='hot')
        plt.title(f'CAM for {model_name} - Normal')
        # plt.colorbar()
        full_path_cam = os.path.join(
            cam_results_path,
            f'CAM_{model_name}_normal.JPG')
        plt.savefig(full_path_cam)

        # CAM for pneumonia image
        plt.imshow(pneumonia_cam, cmap='jet')
        plt.title(f'CAM for {model_name} - Pneumonia')
        # plt.colorbar()
        full_path_cam2 = os.path.join(
            cam_results_path,
            f'CAM_{model_name}_pneumonia.JPG')
        plt.savefig(full_path_cam2)

        """
        Save OSA-based performance variations 
        """
        normal_prediction = model.predict(normal_image)
        pneumonia_prediction = model.predict(pneumonia_image)

        results.append([
            model_name, 'Normal', normal_prediction[0][0]])

        results.append([
            model_name, 'Pneumonia', pneumonia_prediction[0][1]])


evaluate_each_model()


"""
Save OSA-based performance variations 
"""
results_df = pd.DataFrame(results, columns=['Model', 'Class', 'Prediction'])

directory = osa_results_path
file_path = os.path.join(
    directory, "occlusion_sensitivity_analysis_results.csv")

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Create file if it doesn't exist
if not os.path.exists(file_path):
    with open(file_path, 'w') as file:
        file.write("")  # Create an empty file


results_df.to_csv(file_path, index_label=False, index=False)


"""
Plot OSA-based performance variations 
"""


def plot_performance_variations():
    results_df.pivot(
        index='Model', columns='Class',
        values='Prediction').plot(kind='bar')

    plt.figure(figsize=(20, 20))

    plt.ylabel('Prediction')

    plt.title('OSA-based Model Performance Variations')

    full_path_osa = os.path.join(
        osa_results_path,
        f'model_performance_variations.png')

    plt.savefig(full_path_osa)


plot_performance_variations()
