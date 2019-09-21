import random
from keras import models
from keras import layers
from module_evaluate.reference_model import *
from utilities import *


def get_output_ml(path_folder_model, path_data_test):
    list_y_predict = []

    list_path_file_model = get_all_path_file_in_folder(path_folder_model)
    for e_path_model in list_path_file_model:
        y_predict_ml = get_predict_ml(path_data_test, e_path_model)
        list_y_predict.append(y_predict_ml)

    return list_y_predict


def get_output_dl(list_model, path_data_test):
    list_y_predict = []
    for e_model in list_model:
        path_save_model = e_model['path_save_model']
        path_model_checkpoint = e_model['path_model_checkpoint']
        type_model = e_model['type_model']
        y_predict_dl = get_predict_dl(path_data_test, path_save_model, path_model_checkpoint, type_model)

        list_y_predict.append(y_predict_dl)
    return list_y_predict


def ann_model_layer_2():
    # Neural Networks Model
    random.seed(123)
    model = models.Sequential()
    model.add(layers.Dense(8, input_dim=2, kernel_initializer='normal', activation='relu'))  # 2 hidden layers
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def stacking_model(list_model_dl, path_folder_model_ml, path_data_test, y_test=None):
    list_y_ml = get_output_ml(path_folder_model_ml, path_data_test)
    list_y_dl = get_output_dl(list_model_dl, path_data_test)

    list_output_predict_all = list_y_ml + list_y_dl

    model_ann = ann_model_layer_2()
    model_ann.fit(list_output_predict_all, y_test, epochs=20, batch_size=32)


if __name__ == '__main__':

    list_model_dl = [
        {
            "path_save_model": "",
            "path_model_checkpoint": "",
            "type_model": ""
        },
        {
            "path_save_model": "",
            "path_model_checkpoint": "",
            "type_model": ""
        },
        {
            "path_save_model": "",
            "path_model_checkpoint": "",
            "type_model": ""
        },
    ]
    path_folder_ml = ""
    path_data_test = ""
    y_test = ""
    stacking_model(list_model_dl, path_folder_ml, path_data_test, y_test)
