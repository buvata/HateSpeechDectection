import nltk
import time
import random
from random import shuffle
from module_dataset.preprocess_data.handle_text import *
from textblob import TextBlob
from utilities import *


def remove_all_tone(text):
    text = remove_accent(text)
    return text


def random_remove_tone(text, thresh_hold_active=0.3):
    arr_text = text.split()
    n_arr_text = []

    for e_token in arr_text:
        prob = random.random()
        if prob < thresh_hold_active:
            e_token = remove_accent(e_token)
        n_arr_text.append(e_token)

    return " ".join(n_arr_text)


def random_delete_word(text, list_word_exception, thresh_hold_active=0.1):
    arr_text = text.split()
    n_arr_text = []

    for e_token in arr_text:
        prob = random.random()
        if prob < thresh_hold_active and e_token not in list_word_exception:
            pass
        else:
            n_arr_text.append(e_token)

    return " ".join(n_arr_text)


def random_change_synonym_word(text, dict_word_synonym, thresh_hold_active=0.2):

    for e_key, e_value in dict_word_synonym.items():
        prob = random.random()
        if e_key in text and prob < thresh_hold_active:
            shuffle(e_value)
            text = text.replace(e_key, e_value[0])
    return text


def back_translate_data(text):
    try:
        nltk.set_proxy('http://s5.cyberspace.vn:3128')
        blob = TextBlob(text)
        str_en_translate = blob.translate(from_lang='vi', to='en')
        time.sleep(1)

        n_blob = TextBlob(str(str_en_translate))
        str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')
        time.sleep(1)
    except:
        return None

    return str_vn_back_translate


def process_augment_data(text,
                         path_dict_synonym,
                         n_augment_per_sent=5):

    dict_synonym = get_dict_synonym(path_dict_synonym)

    list_word_exception = []
    for e_key, e_value in dict_synonym.items():
        list_word_exception.append(e_key)
        for e_word in list(e_value):
            list_word_exception.append(e_word)

    list_text_augment = []

    text_remove_all_tone = remove_all_tone(text)
    list_text_augment.append(text_remove_all_tone)

    text_del_word = random_delete_word(text, list_word_exception)
    list_text_augment.append(text_del_word)

    text_remove_tone = random_remove_tone(text)
    list_text_augment.append(text_remove_tone)

    text_change_synonym = random_change_synonym_word(text, dict_synonym)
    list_text_augment.append(text_change_synonym)

    for i in range(n_augment_per_sent):
        prob = random.random()

        text_process = text
        if prob < 0.15:
            text_process = random_delete_word(text_process, list_word_exception)

        prob = random.random()
        if prob < 0.15:
            text_process = random_remove_tone(text_process, thresh_hold_active=0.1)

        prob = random.random()
        if prob < 0.15:
            text_process = random_change_synonym_word(text_process, dict_synonym, thresh_hold_active=0.15)

        list_text_augment.append(text_process)

    return list(set(list_text_augment))


def process_augment_data_hate_speech(text,
                                     path_exception_list,
                                     path_synonym,
                                     n_duplicate_sent=2,
                                     n_augment_per_sent=5):

    list_word_exception = get_list_from_file(path_exception_list)
    dict_synonym = get_dict_synonym(path_synonym)
    list_sent_augment = []

    text_delete_word = random_delete_word(text, list_word_exception)
    list_sent_augment.append(text_delete_word)

    for i in range(n_augment_per_sent):
        prob = random.random()
        if prob < 0.10:
            text = random_delete_word(text, list_word_exception)
        if prob < 0.2:
            text = random_change_synonym_word(text, dict_synonym)

        list_sent_augment.append(text)

    list_sent_augment = list(set(list_sent_augment))
    list_sent_augment += [text] * n_duplicate_sent
    return list_sent_augment


if __name__ == '__main__':
    text = "kiểu dáng đẹp nhưng chất lượng và cách may quá không được thời gian giao hàng rất nhanh"
    # print(back_translate_data(text))
    path_1 = "../dataset/support_data/exception_word.csv"
    path_2 = "../dataset/support_data/dict_synonym.csv"
    # lt = process_augment_data(text, path_2)
    # print(lt)
    lt = process_augment_data_hate_speech("cái ảnh cà khịa vl :)", path_1, path_2)
    print(lt)

