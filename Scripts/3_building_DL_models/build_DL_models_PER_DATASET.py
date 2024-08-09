from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D,
                          Dropout, GlobalAveragePooling2D)


"""
Custom implementation of AlexNet
"""


def create_alexnet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


"""
Define all the paths to all the datasets
"""
root_path_to_datasets = "C:/Users/HP/PycharmProjects/MPH/1data/"
d_1 = root_path_to_datasets + "none/d(1)/"
d_2 = root_path_to_datasets + "none/d(2)/"
d_3 = root_path_to_datasets + "none/d(3)/"
d_4 = root_path_to_datasets + "none/d(4)/"

g_D_1 = root_path_to_datasets + "geometric/G-d(1)/"
g_D_2 = root_path_to_datasets + "geometric/G-d(2)/"
g_D_3 = root_path_to_datasets + "geometric/G-d(3)/"
g_D_4 = root_path_to_datasets + "geometric/G-d(4)/"

n_D_1 = root_path_to_datasets + "gaussianNoise/GN-d(1)/"
n_D_2 = root_path_to_datasets + "gaussianNoise/GN-d(2)/"
n_D_3 = root_path_to_datasets + "gaussianNoise/GN-d(3)/"
n_D_4 = root_path_to_datasets + "gaussianNoise/GN-d(4)/"

r_D_1 = root_path_to_datasets + "randomColorSpace/RCS-d(1)/"
r_D_2 = root_path_to_datasets + "randomColorSpace/RCS-d(2)/"
r_D_3 = root_path_to_datasets + "randomColorSpace/RCS-d(3)/"
r_D_4 = root_path_to_datasets + "randomColorSpace/RCS-d(4)/"

"""
Define all the paths to all the models
"""
root_path_to_models = "C:/Users/HP/PycharmProjects/MPH/2models/"
none = root_path_to_models + "_none/"
geometric = root_path_to_models + "geometric/"
noise_gaussian = root_path_to_models + "noise_gaussian/"
random_color_space = root_path_to_models + "random_color_space/"

"""
Set the image size for each model 
"""
image_size_for_each_model = {
    'AlexNet': (227, 227),
    'MobileNet': (224, 224), }

"""
Set the number of classes and model hyperparameters
"""
number_of_classes = 2
batch_size = 32
epochs = 10
learning_rate = 0.001

"""
Create the image data generator for data ingestion
"""
training_data_generator = ImageDataGenerator(rescale=1./255)

"""
Initialize DL models in dictionaries
"""
# define the models
alexNet_and_MobileNet_models = {
    'AlexNet': create_alexnet(
        input_shape=(227, 227, 3),
        num_classes=number_of_classes),
    'MobileNet': MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)), }

"""
Train and save all 4 models given a dataset
"""


def train_and_save_models(
        path_to_dataset, model_name_prefix,
        path_to_model_group, dl_models):
    # load the training data
    training_data = training_data_generator.flow_from_directory(
        path_to_dataset,
        target_size=image_size_for_each_model['AlexNet'],
        batch_size=batch_size,
        class_mode='categorical')

    # compile and fit the models
    for model_name, model in dl_models.items():
        if model_name in ['MobileNet', 'VGG16']:
            x = model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(number_of_classes, activation='softmax')(x)
            model = Model(inputs=model.input, outputs=x)

        model.compile(
            optimizer=Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(
            training_data,
            steps_per_epoch=training_data.samples // batch_size,
            epochs=epochs)

        full_path = path_to_model_group + f"{model_name_prefix}_{model_name}.h5"
        model.save(full_path)


# train and save models
train_and_save_models(
    path_to_dataset=r_D_4,
    model_name_prefix='RCS-D(4)',
    path_to_model_group=random_color_space,
    dl_models=alexNet_and_MobileNet_models)
