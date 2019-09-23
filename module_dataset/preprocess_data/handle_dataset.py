import sentencepiece as spm
from module_train.ml_model.feature_extract import extract_feature
from module_dataset.preprocess_data.handle_data_augmentation import *

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
from gensim.models import FastText


def get_data_contest(path_data, path_preprocess_data, path_label=None):

    a = []
    l_full_sample = []

    if path_label is not None:
        dict_label_id = {}
        regex = 'train_'
        df = pd.read_csv(path_label, sep=",", names=["id", "label"])
        list_id = df.id.tolist()
        list_label = df.label.tolist()
        for i in range(len(list_id)):
            dict_label_id[list_id[i]] = list_label[i]

    else:
        regex = 'test_'

    with open(path_data, 'r') as file:
        for line in file:
            if regex in line:
                l_full_sample.append(a)
                a = [line]
            elif line != '\n':
                a.append(line)

    l_full_sample.append(a)

    with open(path_preprocess_data, "w") as wf:
        for e_sample in l_full_sample[1:]:
            full_line = " ".join(e_sample).replace("\n", "").replace("|", ",")

            id_data = full_line.split(",")[0]
            data = ",".join(full_line.split(",")[1:])

            if data[0] == '"' and data[-1] == '"':
                data = data[1: -1]

            if path_label is None:
                line_write_file = "{}|{}\n".format(id_data, data)
            else:
                line_write_file = "{}|{}\n".format(data, dict_label_id[id_data])
            print(line_write_file)
            wf.write(line_write_file)


def handle_data_with_punc_emoji_space(path_data_process_raw, path_data_after_process, is_train=True):
    with open(path_data_after_process, "w") as wf:
        with open(path_data_process_raw, "r") as rf:
            for e_line in rf.readlines():
                e_line = e_line.replace("\n", "").split("|")
                if is_train:
                    text = e_line[0]
                    id_data = e_line[1]
                    text = handle_text_hate_speech(text)
                    line_write = "{}|{}\n".format(text, id_data)
                else:
                    id_data = e_line[0]
                    text = e_line[1]
                    text = handle_text_hate_speech(text)
                    line_write = "{}|{}\n".format(id_data, text)

                wf.write(line_write)


def make_corpus_data(path_file_train, path_corpus, is_train=True):
    with open(path_corpus, 'a') as wf:
        with open(path_file_train, 'r') as rf:
            for e_line in rf.readlines():
                if is_train:
                    text = e_line.split("|")[0].replace("\n", "")
                else:
                    text = e_line.split("|")[1].replace("\n", "")

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
        for e_line in rf.readlines():
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


def train_embedding_fasttext(cf):
    path_corpus = cf['path_corpus_data']
    l_split_lines = []
    with open(path_corpus, "r") as r_corpus:
        for e_line in r_corpus.readlines():
            e_line = e_line.replace("\n", "")
            e_line = e_line.replace("\r", "")
            split_line = e_line.split(" ")
            l_split_lines.append(split_line)

    ft = FastText(l_split_lines, sg=1, iter=5, min_n=2, min_count=2, size=200)
    print(ft.wv.most_similar("vl", topn=20))
    ft.wv.save_word2vec_format("social_embedding_200.txt", binary=False)


def making_exception_list_kw_from_synonym(cf):
    path_synonym_data = cf['path_synonym']
    path_exception_word = cf['path_exception_word']

    with open(path_exception_word, "w") as wf:
        list_full_token = []
        with open(path_synonym_data, "r") as rf:
            for e_line in rf.readlines():
                e_line = e_line.replace(", ", ",").replace("\n", ",")
                e_line_no_accent = remove_accent(e_line)
                full_line = e_line + e_line_no_accent
                list_token_synonym = full_line.split(",")
                list_full_token += list_token_synonym

        for e_token in list(set(list_full_token)):
            wf.write("{}\n".format(e_token))


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

    # path_file_train_origin = "../preprocess_data/normal.csv"
    # path_file_train_augment = "../dataset/data_for_train/exp_augment.csv"
    # path_dict_synonym = "../dataset/support_data/dict_synonym.json"
    # make_data_with_augmentation(path_file_train_origin, path_file_train_augment,
    #                             path_dict_synonym,
    #                             n_augment_per_sent=5)
    path_cf = "/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/preprocess_data/config_dataset.json"
    cf = load_config(path_cf)
    path_raw_data = cf['train_raw_text']
    path_preprocess_data = cf['test_preprocess_text']
    path_process_punct = cf['test_process_emoji_punct']
    path_label = cf['train_label']
    # get_data_contest(path_raw_data, path_preprocess_data, path_label)
    # handle_data_with_punc_emoji_space(path_preprocess_data, path_process_punct, is_train=False)
    # split_train_test(cf['train_process_emoji_punct'],
    #                  cf['path_folder_save_data_for_dl'],
    #                  cf['name_train'],
    #                  cf['name_test'])
    # make_corpus_data(cf['test_process_emoji_punct'], cf['path_corpus_data'], is_train=False)
    # train_embedding_fasttext(cf)
    making_exception_list_kw_from_synonym(cf)
