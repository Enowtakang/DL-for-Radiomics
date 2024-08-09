import os
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve)


"""
Set paths
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

# path to test dataset
path_to_test_dataset = "C:/Users/HP/PycharmProjects/MPH/datasets/test/"

# paths to results
root_path_to_results = "C:/Users/HP/PycharmProjects/MPH/4results/"
numerical_results = root_path_to_results + "1_numerical_evaluations/"
roc_auc_curves = root_path_to_results + "2_roc_auc_curves/"
confusion_matrices = root_path_to_results + "3_confusion_matrices/"


"""
Make a dictionary of models
"""
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
    'N-D(4)_AlexNet': N_D_4_AlexNet, 'N-D(4)-MobileNet': N_D_4_MobileNet,

    'R-D(1)-AlexNet': R_D_1_AlexNet, 'R-D(1)-MobileNet': R_D_1_MobileNet,
    'R-D(2)-AlexNet': R_D_2_AlexNet, 'R-D(2)-MobileNet': R_D_2_MobileNet,
    'R-D(3)-AlexNet': R_D_3_AlexNet, 'R-D(3)-MobileNet': R_D_3_MobileNet,
    'R-D(4)-AlexNet': R_D_4_AlexNet, 'R-D(4)-MobileNet': R_D_4_MobileNet, }


"""
Set the image size for each model
"""
image_size_for_each_model = {
    'AlexNet': (227, 227),
    'MobileNet': (224, 224), }

"""
Load the testing dataset
"""
batch_size = 13

test_datagen = ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_directory(
    path_to_test_dataset,
    target_size=image_size_for_each_model['AlexNet'],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

"""
Initialize results storage
"""
results = list()

"""
Evaluate each model
"""


def evaluate_each_model():
    for model_name, path_to_model in paths_to_models_dictionary.items():
        """ 
        load the model
        """
        model = load_model(path_to_model)

        """ 
        predict on the test dataset
        """
        predictions = model.predict(test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)

        """
        get the true classes
        """
        true_classes = test_dataset.classes
        class_labels = list(test_dataset.class_indices.keys())

        """
        compute metrics
        """
        accuracy = accuracy_score(
            true_classes, predicted_classes)
        recall = recall_score(
            true_classes, predicted_classes, average='weighted')
        precision = precision_score(
            true_classes, predicted_classes, average='weighted')
        f1 = f1_score(
            true_classes, predicted_classes, average='weighted')
        cohen_kappa = cohen_kappa_score(
            true_classes, predicted_classes)

        """
        confusion matrix for specificity and negative predicted value
        """
        cm = confusion_matrix(true_classes, predicted_classes)
        tn = cm[0, 0]  # True Negatives
        fp = cm[0, 1]  # False Positives
        fn = cm[1, 0]  # False Negatives
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = (tn / (tn + fn)) if (tn + fn) > 0 else 0  # Negative Predictive Value

        """
        ROC-AUC
        """
        if len(class_labels) == 2:  # Binary classification
            roc_auc = roc_auc_score(
                true_classes, predictions[:, 1])  # Assuming class 1 is positive
            fpr, tpr, thresholds = roc_curve(
                true_classes, predictions[:, 1])
        else:
            roc_auc = None  # Not applicable for multi-class
            fpr, tpr = None, None

        """
        Save the numerical evaluation metrics
        """
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'Specificity': specificity,
            'Negative Predictive Value': npv,
            'Cohen’s Kappa': cohen_kappa,
            'ROC-AUC': roc_auc
        })

        """
        Plot ROC Curve
        """
        if roc_auc is not None:
            plt.figure()
            plt.plot(
                fpr, tpr, color='blue',
                label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(
                roc_auc_curves, f'roc_curve_{model_name}.JPG'))
            plt.close()

        # Annotated Confusion Matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.savefig(os.path.join(
            confusion_matrices, f'confusion_matrix_{model_name}.JPG'))
        plt.close()


evaluate_each_model()


"""
Convert results to dataframe and save results to CSV
"""
# convert results to dataframe
results_df = pd.DataFrame(results)

directory = numerical_results
file_path = os.path.join(directory, "numerical_results.csv")

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Create file if it doesn't exist
if not os.path.exists(file_path):
    with open(file_path, 'w') as file:
        file.write("")  # Create an empty file


results_df.to_csv(file_path, index_label=False, index=False)










