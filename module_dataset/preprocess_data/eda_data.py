from module_dataset.preprocess_data.handle_text import *
from module_train.ml_model.get_important_kw import *

from utilities import *
from pyvi import ViTokenizer


def get_important_kw(path_data_train, path_save_important_kw, n_kw_extract=50):
    list_train_data = []
    list_label = []
    with open(path_data_train, "r") as rf:
        for e_line in rf.readlines():
            arr_e_line = e_line.replace("\n", "").split("|")

            text = handle_text_before_make_piece(arr_e_line[0])
            text = ViTokenizer.tokenize(text)

            list_train_data.append(text)
            list_label.append(arr_e_line[1])

    get_feature_importance(list_train_data, list_label, path_save_important_kw, n_kw_extract)





if __name__ == '__main__':
    path_config = "/home/trangtv/Documents/project/TextClassification/module_dataset/preprocess_data/" \
                  "config_dataset.json"
    cf = load_config(path_config)
    get_important_kw(cf['path_data_raw_train'], cf['path_save_important_kw'], 100)
