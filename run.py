from module_dataset.preprocess_data.handle_dataset import *
from module_dataset.preprocess_data.hanlde_dataloader import *
from module_dataset.preprocess_data.handle_text import *
from module_dataset.preprocess_data.eda_data import *
from module_dataset.preprocess_data.handle_data_augmentation import *
from module_train.ml_model.model_ml import *
from module_train.train_ml import *

if __name__ == '__main__':

    path_file_raw = "module_dataset/dataset/raw_data/train.csv"
    path_corpus = "module_dataset/dataset/raw_data/corpus_data.csv"

    # split train,test data
    # get train, test feature for ml
    path_save_data = "module_dataset/dataset/data_for_train/"
    name_train = "exp_train.csv"
    name_test = "exp_test.csv"

    # path_file_train_augment = "module_dataset/dataset/data_for_train/exp_train.csv"
    # path_file_test_augment = "module_dataset/dataset/data_for_train/exp_train.csv"
    path_file_train = "module_dataset/dataset/data_for_train/exp_train.csv"
    path_file_test = "module_dataset/dataset/data_for_train/exp_test.csv"
    path_save_model_extract_ft = "module_dataset/dataset/data_for_train/extract_ft.pkl"
    path_save_pickle_data_train = "module_train/save_model/model_ml/data_train_ft.pkl"
    path_save_pickle_data_test = "module_train/save_model/model_ml/data_test_ft.pkl"

    path_normal_dataset = "module_dataset/dataset/data_for_train/data_train.csv"
    norm_data_format(path_file_raw, path_normal_dataset)

    split_train_test(path_normal_dataset, path_save_data, name_train, name_test, test_size=0.2)

    # augment data
    path_dict_synonym = "module_dataset/dataset/support_data/dict_synonym.json"
    lt = process_augment_data(path_file_train, path_dict_synonym)

    # get feature tfidf for ml
    make_dataset_for_ml(path_file_train, path_save_pickle_data=path_save_pickle_data_train,
                     path_save_model_extract_ft=path_save_model_extract_ft, is_train=True)

    make_dataset_for_ml(path_file_test, path_save_pickle_data=path_save_pickle_data_test,
                     path_save_model_extract_ft=path_save_model_extract_ft, is_train=False)

    # get importance word
    path_save_important_kw = 'module_dataset/dataset/support_data/important_kw.csv'
    get_important_kw(path_file_train, path_save_important_kw, n_kw_extract=50)

    # train ml model
    path_save_model_ml = "../module_train/save_model/model_ml"
    path_data_ml = "../module_train/save_model/model_ml"
    name_train_ml = "data_train_ft.pkl"
    train_all_ml_model(path_save_model_ml, path_data_ml, name_train_ml, name_test=None)

    # predict test data















