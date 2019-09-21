from module_dataset.preprocess_data.handle_text import *
from module_train.ml_model.get_important_kw import *

from utilities import *


def get_important_kw(path_data_train, path_save_important_kw, n_kw_extract=50):
    list_train_data = []
    list_label = []
    with open(path_data_train, "r") as rf:
        for e_line in rf.readlines():
            arr_e_line = e_line.replace("\n", "").split("|")

            # text = handle_text_before_make_piece(arr_e_line[0])
            # text = ViTokenizer.tokenize(text)
            text = arr_e_line[0].lower()
            list_train_data.append(text)
            list_label.append(arr_e_line[1])

    get_feature_importance(list_train_data, list_label, path_save_important_kw, n_kw_extract)


def get_data_type_1_2(path_dataset, path_offensive, path_hate):
    w_offensive = open(path_offensive, "w")
    w_hate = open(path_hate, "w")
    with open(path_dataset, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            text_data = e_line.split("|")[0]
            label = int(e_line.split("|")[1])
            if label == 1:
                w_offensive.write(text_data + "\n")
            if label == 2:
                w_hate.write(text_data + "\n")

    w_offensive.close()
    w_hate.close()

if __name__ == '__main__':
    path_config = "/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/preprocess_data/" \
                  "config_dataset.json"
    cf = load_config(path_config)
    # get_important_kw(cf['train_process_emoji_punct'], cf['path_save_important_kw'], 200)
    get_data_type_1_2(cf['train_process_emoji_punct'],
                      cf['path_offensive_data'],
                      cf['path_hate_data'])
