from module_dataset.preprocess_data.hanlde_dataloader import load_data_ml
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
from module_train.ml_model.model_ml import *

import matplotlib.pyplot as plt

import pandas as pd
import os


def train_using_each_ml(model, params, path_save_model, name_model, path_data, name_train, name_test=None):
    
    dict_result = {}
    path_file_train = os.path.join(path_data, name_train)
    x_train, y_train = load_data_ml(path_file_train)

    if name_test is not None:
        path_file_test = os.path.join(path_data, name_test)
        x_test, y_test = load_data_ml(path_file_test)

    grid_search = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=1, verbose=1)

    grid_search.fit(x_train, y_train)

    path_save_model = os.path.join(path_save_model, name_model)
    joblib.dump(grid_search.best_estimator_, path_save_model)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")

    dict_result['accuracy_train'] = grid_search.best_score_

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    if name_test is not None:
        score = grid_search.score(x_test, y_test)
        print("accucary in test: {}".format(score))
        dict_result['accuracy_test'] = score
    print(dict_result)

    return dict_result


def train_all_ml_model(path_save_model, path_data, name_train, name_test=None):

    models = [
        model_svm(),
        model_random_forest(),
        model_xgboost(),
        model_lgbm(),
        model_gradient_boosting()
    ]

    list_name = []
    list_acc_train = []
    list_acc_test = []
    for e_model in models:
        model, params, name = e_model
        list_name.append(name)
        dict_result = train_using_each_ml(model, params, path_save_model, name, path_data, name_train, name_test)

        list_acc_train.append(np.round(dict_result['accuracy_train'], 2))

        if name_test is not None:
            list_acc_test.append(np.round(dict_result['accuracy_test']))

    if name_test is not None:
        report = list(zip(list_name, list_acc_train, list_acc_test))
        cv_df = pd.DataFrame(report, columns=['model_name', 'accuracy_train', 'accuracy_test'])
    else:
        report = list(zip(list_name, list_acc_train))
        cv_df = pd.DataFrame(report, columns=['model_name', 'accuracy_train'])

    name_report_result_ml = os.path.join(path_save_model, "report_all.csv")
    cv_df.to_csv(name_report_result_ml, index=False)

    return cv_df


def load_model_ml(path_save_model):
    model_ml = joblib.load(path_save_model)
    return model_ml


if __name__ == "__main__":
    path_save_model = "/home/taibk/Documents/Code_ML/HateSpeechDectection/module_train/save_model/model_ml"
    path_data = "/home/taibk/Documents/Code_ML/HateSpeechDectection/module_train/save_model/model_ml"
    name_train = "data_train_ft.pkl"
    train_all_ml_model(path_save_model, path_data, name_train, name_test=None)