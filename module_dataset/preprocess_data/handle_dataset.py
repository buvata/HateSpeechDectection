import sentencepiece as spm
from module_train.ml_model.feature_extract import extract_feature
from module_dataset.preprocess_data.handle_data_augmentation import *

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pickle
import os


def make_corpus_data(path_file_train, path_corpus):
    with open(path_corpus, 'a') as wf:
        with open(path_file_train, 'r') as rf:
            for e_line in rf.readlines():
                text = e_line.split("|")[2].replace("\n", "")

                text = handle_text_before_make_piece(text)
                wf.write(text + "\n")

                text = remove_accent(text)
                wf.write(text + "\n")


def norm_data_format(path_file_data, path_data_norm):
    with open(path_data_norm, "w") as wf:
        with open(path_file_data, "r") as rf:
            for e_line in rf.readlines():
                arr_line = e_line.split("|")
                if len(arr_line) > 3:
                    print("error")
                    print(e_line)
                text_data = arr_line[2].replace("\n", "")
                label = arr_line[1]
                line_write = "{}|{}\n".format(text_data, label)
                wf.write(line_write)


def build_sentence_piece(path_corpus, path_save_model, vocab_size, shuffle_input_sentence='true'):
    command_to_train = '--input={} --model_prefix={} ' \
                       '--vocab_size={} --shuffle_input_sentence={} ' \
                       '--hard_vocab_limit=false'.format(path_corpus,
                                                                            path_save_model,
                                                                            vocab_size,
                                                                            shuffle_input_sentence)

    spm.SentencePieceTrainer.Train(command_to_train)


def convert_data_with_piece(path_file_data_origin, path_file_data_piece, path_model_piece):
    s = spm.SentencePieceProcessor()
    s.Load(path_model_piece)

    with open(path_file_data_piece, 'a') as wf:
        with open(path_file_data_origin, 'r') as rf:
            for e_line in rf.readlines():
                arr_e_line = e_line.split("|")
                text = arr_e_line[2].replace("\n", "")
                label = arr_e_line[1]

                text = handle_text_before_make_piece(text)
                text = norm_text_with_sub_word(text, s)

                wf.write(text + "|" + label + "\n")


def make_data_with_augmentation(path_file_train_origin, path_file_train_augment,
                                path_dict_synonym,
                                n_augment_per_sent=10):

    with open(path_file_train_augment, 'a') as wf:
        with open(path_file_train_origin, 'r') as rf:
            for e_line in rf.readlines():
                e_line = e_line.replace("\n", "")
                list_augment_text = process_augment_data(e_line,
                                                         path_dict_synonym,
                                                         n_augment_per_sent)

                for e_augment_text in list_augment_text:
                    wf.write(e_augment_text + "\n")


def split_train_test(path_file_train, path_save_data, name_train, name_test, test_size=0.2):
    list_x_full = []
    list_y_full = []

    with open(path_file_train, 'r') as rf:
        for e_line in rf.readlines()[1:]:
            arr_line = e_line.replace("\n", "").split("|")
            list_x_full.append(arr_line[0])
            list_y_full.append(arr_line[1])

    x_train, x_test, y_train, y_test = train_test_split(list_x_full, list_y_full,
                                                        test_size=test_size,
                                                        stratify=list_y_full)

    path_data_train = os.path.join(path_save_data, name_train)
    path_data_test = os.path.join(path_save_data, name_test)

    with open(path_data_train, "w") as wf_train:
        for i in range(len(x_train)):
            line_write = "{}|{}\n".format(x_train[i], y_train[i])
            wf_train.write(line_write)

    with open(path_data_test, "w") as wf_test:
        for i in range(len(x_test)):
            line_write = "{}|{}\n".format(x_test[i], y_test[i])
            wf_test.write(line_write)


def make_dataset_for_ml(path_file_train_augment, path_save_pickle_data,
                        path_save_model_extract_ft=None, is_train=True):
        clf_extract_feature = extract_feature()

        list_text_data = []
        list_label = []
        with open(path_file_train_augment, "r") as rf:
            for e_line in rf.readlines():
                arr_line = e_line.replace("\n", "").split("|")
                list_text_data.append(arr_line[0])
                if is_train:
                    list_label.append(arr_line[1])

        if is_train and path_save_model_extract_ft is not None:
            clf_extract_feature.fit_transform(list_text_data)
            joblib.dump(clf_extract_feature, path_save_model_extract_ft)
        else:
            clf_extract_feature = joblib.load(path_save_model_extract_ft)

        x_train = clf_extract_feature.transform(list_text_data)

        with open(path_save_pickle_data, "wb") as wf:
            if is_train:
                pickle.dump([x_train, list_label], wf)

            else:
                pickle.dump([x_train], wf)


if __name__ == '__main__':
    '''
    path_file_train = "../dataset/raw_data/train_data.csv"
    path_corpus = "../dataset/raw_data/corpus_data.csv"
    # path_model_piece = "../dataset/support_data/model_piece"
    # make_corpus_data(path_file_train, path_corpus)
    # build_sentence_piece(path_corpus, path_model_piece, 20000)

    path_save_data = "../dataset/data_for_train/"
    name_train = "exp_train.csv"
    name_test = "exp_test.csv"

    path_file_train_augment = "../dataset/data_for_train/exp_train.csv"
    path_save_model_extract_ft = "../dataset/data_for_train/extract_ft.pkl"
    path_save_pickle_data = "../../module_train/save_model/model_ml/data_ft.pkl"

    split_train_test(path_file_train, path_save_data, name_train, name_test, test_size=0.2)

    make_dataset_for_ml(path_file_train_augment, path_save_pickle_data,
                        path_save_model_extract_ft=path_save_model_extract_ft, is_train=True)
    '''

    # norm_data_format("../dataset/data_for_train/exp_augment_train.csv", "normal.csv")

    path_file_train_origin = "../preprocess_data/normal.csv"
    path_file_train_augment = "../dataset/data_for_train/exp_augment.csv"
    path_dict_synonym = "../dataset/support_data/dict_synonym.json"
    make_data_with_augmentation(path_file_train_origin, path_file_train_augment,
                                path_dict_synonym,
                                n_augment_per_sent=5)