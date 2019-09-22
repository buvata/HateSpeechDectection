import re
import unicodedata
from utilities import *
import string


# make global set punctuation
set_punctuations = set(string.punctuation)
list_punctuations_out = ['”', '”', "›", "“"]
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)


dict_typing = get_dict_typing_error("/home/taibk/Documents/Code_ML/HateSpeechDectection/module_dataset/dataset/support_data/typing_error_telex.csv")


patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}
number_token = "<number>"


def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\xa0', ' ')
    return text


def remove_accent(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output


def fix_typing_error(text, dict_typing_error):
    for e_key, e_value in dict_typing_error.items():
        if e_key in text:
            text = text.replace(e_key, e_value)

    return text


def remove_multi_space(text):
    text = text.replace("\t", " ")
    text = re.sub("\s\s+", " ", text)
    # handle exception when line just all of punctuation
    if len(text) == 0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text) == 0:
        pass
    else:
        if text[-1] == " ":
            text = text[:-1]

    return "".join(text)


def handle_punctuation(text):
    # need replace | for split field in csv file

    l_new_char = []
    for e_char in text:

        if e_char not in list(set_punctuations):
            l_new_char.append(e_char)
        else:
            l_new_char.append(" {} ".format(e_char))
    text = "".join(l_new_char)

    return text


def norm_text_with_sub_word(text, s, convert_number=True):
    n_arr = []
    arr_text_sub = s.EncodeAsPieces(text)

    for e_arr in arr_text_sub:
        if convert_number:
            if e_arr[0] == "▁":
                if not e_arr[1:].isdigit():
                    n_arr.append(e_arr[1:])
                else:
                    n_arr.append(number_token)
            else:
                if not e_arr.isdigit():
                    n_arr.append("##{}".format(e_arr))
                else:
                    n_arr.append("##{}".format(number_token))
        else:
            if e_arr[0] == "▁":
                n_arr.append(e_arr[1:])
            else:
                n_arr.append("##{}".format(e_arr))

    return " ".join(n_arr).replace("\n", "")


def handle_text_before_make_piece(text):
    text = normalize_text(text)
    text = handle_punctuation(text)
    text = fix_typing_error(text, dict_typing)
    text = remove_multi_space(text)
    return text


if __name__ == '__main__':
    '''
    text_test = "hôm nay cos ver không tố lăms"
    dict_typing = get_dict_typing_error("../module_dataset/dataset/support_data/typing_error_telex.csv")
    print(fix_typing_error(text_test, dict_typing))
    '''