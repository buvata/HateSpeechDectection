import time
import random
from random import shuffle
from module_dataset.preprocess_data.handle_text import *
from textblob import TextBlob
from utilities import *

list_alphabet = list(string.ascii_lowercase)


def get_dict_synonym(path_file_dict):
    dict_sym = {}
    with open(path_file_dict, "r") as rf:
        for e_line in rf.readlines():
            e_line = e_line.replace(", ", ",").replace("\n", "")
            e_new_line = remove_all_tone(e_line)
            arr_line = e_line.split(",")
            arr_rm_tone_line = e_new_line.split(",")

            # use for replace sure key wrap by space
            # tmp = []
            # for e_token in arr_line:
            #     tmp.append(" {} ".format(e_token))
            # make array like capitalize
            arr_line_capitalize = []
            for e_token in arr_line:
                arr_line_capitalize.append(e_token.capitalize())

            for e_token_1 in arr_line:
                n_tmp = arr_line.copy()
                n_tmp.remove(e_token_1)
                if " " == e_token_1[0]:
                    pass
                else:
                    for e_token_sub_1 in n_tmp:
                        if " " == e_token_sub_1[0]:
                            n_tmp.remove(e_token_sub_1)
                dict_sym[e_token_1] = n_tmp

            for e_token_2 in arr_line_capitalize:
                n_tmp_capitalize = arr_line_capitalize.copy()
                n_tmp_capitalize.remove(e_token_2)
                if " " == e_token_2[0]:
                    pass
                else:
                    for e_token_sub_2 in n_tmp_capitalize:
                        if " " == e_token_sub_2[0]:
                            n_tmp_capitalize.remove(e_token_sub_2)
                    dict_sym[e_token_2] = n_tmp_capitalize

            for e_token_3 in arr_rm_tone_line:
                n_tmp_3 = arr_rm_tone_line.copy()
                n_tmp_3.remove(e_token_3)
                if " " == e_token_3[0]:
                    pass
                else:
                    for e_token_sub_3 in n_tmp_3:
                        if " " == e_token_sub_3[0]:
                            n_tmp_3.remove(e_token_sub_3)
                dict_sym[e_token_3] = n_tmp_3

    return dict_sym


def get_number_token_with_length(text, postion):
    count = 0
    for i in range(postion):
        if text[i] == " ":
            count += 1
    return count


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


def check_mask_word_in_list_exception(text, list_word_exception):
    text = text.lower()
    list_mask_exception = [0] * len(text.split(" "))
    for e_word_exception in list_word_exception:
        if e_word_exception in text:

            position = text.find(e_word_exception)
            start_number_token = get_number_token_with_length(text, position)
            end_number_token = start_number_token + len(e_word_exception.split(" "))
            for i in range(start_number_token, end_number_token):
                list_mask_exception[i] = 1

    return list_mask_exception


def random_delete_word(text, list_mask_exception, thresh_hold_active=0.1):
    arr_text = text.split(" ")
    n_arr_text = []

    for idx, e_token in enumerate(arr_text):
        prob = random.random()
        if prob < thresh_hold_active and list_mask_exception[idx] != 1:
            pass
        else:
            n_arr_text.append(e_token)

    return " ".join(n_arr_text)


def random_change_synonym_word(text, dict_word_synonym, thresh_hold_active=0.2):
    list_value_must_skip = []
    for e_key, e_value in dict_word_synonym.items():
        prob = random.random()
        if e_key in text and prob < thresh_hold_active and e_key not in list_value_must_skip:
            shuffle(e_value)
            text = text.replace(e_key, e_value[0])
            list_value_must_skip += e_value.copy()
    return text


def random_remove_insert_character(text, thresh_hold_active_char=0.15):
    if len(text) < 3:
        return text

    n_text = []
    count_augment = 0
    for e_character in text:
        prob = random.random()
        if prob < thresh_hold_active_char and count_augment <= 1:
            count_augment += 1
            pass
        else:
            n_text.append(e_character)

        if prob / thresh_hold_active_char < 0.1 and count_augment <= 1:
            n_text.append(e_character)
            shuffle(list_alphabet)
            n_text.append(list_alphabet[0])
            count_augment += 1

    return "".join(n_text)


def random_make_new_word_by_character(text, thresh_hold_active=0.2):
    arr_text = text.split(" ")
    n_arr_text = []
    for e_token in arr_text:
        prob = random.random()
        if prob < thresh_hold_active:
            e_token = random_remove_insert_character(e_token, thresh_hold_active_char=0.1)
        n_arr_text.append(e_token)
    return " ".join(n_arr_text)


def back_translate_data(text):
    try:
        # nltk.set_proxy('http://s5.cyberspace.vn:3128')
        blob = TextBlob(text)
        str_en_translate = blob.translate(from_lang='vi', to='en')
        time.sleep(1)

        n_blob = TextBlob(str(str_en_translate))
        str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')
        print(str_vn_back_translate)
        time.sleep(1)
    except:
        return None

    return str(str_vn_back_translate)


def process_augment_hate_data(text,
                              path_exception_list,
                              path_synonym,
                              n_augment_per_sent=2):
    list_text_augment = []

    list_word_exception = get_list_from_file(path_exception_list)
    dict_synonym = get_dict_synonym(path_synonym)

    text_remove_all_tone = remove_all_tone(text)
    list_text_augment.append(text_remove_all_tone)

    text_remove_tone = random_remove_tone(text)
    list_text_augment.append(text_remove_tone)

    text_random_change_synonym_word = random_change_synonym_word(text, dict_synonym, thresh_hold_active=0.5)
    list_text_augment.append(text_random_change_synonym_word)

    text_lower = text.lower()
    list_text_augment.append(text_lower)

    for i in range(n_augment_per_sent):

        prob = random.random()
        if prob < 0.8:
            text = random_change_synonym_word(text, dict_synonym, thresh_hold_active=1)

        prob = random.random()
        if prob < 0.2:
            list_mask_exception_after_change_synonym = check_mask_word_in_list_exception(text, list_word_exception)
            text = random_delete_word(text, list_mask_exception_after_change_synonym, thresh_hold_active=0.1)

        prob = random.random()
        if prob < 0.05:
            text = random_remove_tone(text, thresh_hold_active=0.05)

        prob = random.random()
        if prob < 0.05:
            text = random_make_new_word_by_character(text, thresh_hold_active=0.05)

        list_text_augment.append(text)

    return list(set(list_text_augment))


if __name__ == '__main__':
    text = " đéo biết tnao cho đúng nữa"
    # print(back_translate_data(text))
    path_1 = "../dataset/support_data/exception_word.csv"
    path_2 = "../dataset/support_data/dict_synonym.csv"
    # lt = process_augment_hate_data(text, path_1, path_2)
    # print(lt)
    dict_synonym = get_dict_synonym(path_2)
    print(dict_synonym)
    for i in range(10):
        a = random_change_synonym_word(text, dict_synonym, thresh_hold_active=1)
        print(a)
