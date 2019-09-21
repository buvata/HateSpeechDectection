from module_dataset.preprocess_data.handle_text import *
from module_train.ml_model.get_important_kw import *
from collections import Counter
from utilities import *
from pyvi import ViTokenizer
import matplotlib.pyplot as plt


def get_important_kw(path_data_train, path_save_important_kw, n_kw_extract=150):
    list_train_data = []
    list_label = []
    with open(path_data_train, "r") as rf:
        for e_line in rf.readlines():
            arr_e_line = e_line.replace("\n", "").split("|")

            text = handle_text_before_make_piece(arr_e_line[0])
            # text = ViTokenizer.tokenize(text)

            list_train_data.append(text)
            list_label.append(arr_e_line[1])

    get_feature_importance(list_train_data, list_label, path_save_important_kw, n_kw_extract)


def ratio_label(path_data):
    data_train_label = pd.read_csv(path_data)
    n_label_0, n_label_1, n_label_2 = 0, 0, 0
    for i in data_train_label.label_id:
        if i == 0:
            n_label_0 += 1
        elif i == 1:
            n_label_1 += 1
        else:
            n_label_2 += 1
    return n_label_0, n_label_1, n_label_2


def len_sent(path_data):
    with open(path_data, 'r') as rf:
        max_len, min_len = 0, 1000
        for e_line in rf.readlines():
            text = e_line.split("|")[0].replace("\n", "")
            if len(text.split(' ')) > max_len:
                max_len = len(text.split(' '))
            if len(text.split(' ')) < min_len:
                min_len = len(text.split(' '))
    return max_len, min_len


def num_len_sent(path_data):
    num_len, len_txt, list_len, values, labels = [], {}, [], [], []
    with open(path_data, 'r') as rf:
        for e_line in rf.readlines():
            text = e_line.split("|")[0].replace("\n", "")
            label = e_line.split("|")[1].replace("\n", "")
            labels.append(label)
            values.append(len(text.split(' ')))

        for key in labels:
            len_txt[key] = []

        for i, value in enumerate(values):
            len_txt[labels[i]].append(value)

    dict0 = Counter(list(len_txt.values())[0])
    dict1 = Counter(list(len_txt.values())[1])
    dict2 = Counter(list(len_txt.values())[2])
    list_len.append(dict0)
    list_len.append(dict1)
    list_len.append(dict2)
    for i in range(3):
        num_sent = list(list_len[i].values())
        len_sent = list(list_len[i].keys())
        plt.bar(len_sent, num_sent)
        plt.xlabel('len')
        plt.ylabel('num_sent')
        plt.title("label_{}".format(i))
        plt.show()




if __name__ == '__main__':
    path_data = "../dataset/raw_data/vn_hate_speech/03_train_label.csv"
    n_label_0, n_label_1, n_label_2 = ratio_label(path_data)
    print("label_0:", n_label_0)
    print("label_1:", n_label_1)
    print("label_2:", n_label_2)

    path_data_train = "../dataset/raw_data/vn_hate_speech/train_process_emoji_punct.csv"
    max, min = len_sent(path_data_train)

    num_len_sent(path_data_train)


    path_save_important_kw = '../dataset/support_data/important_kw.csv'
    get_important_kw(path_data_train, path_save_important_kw, n_kw_extract=200)