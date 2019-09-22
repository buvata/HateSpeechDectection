import re
import unicodedata
from utilities import *
import string
import emoji


# make global set punctuation
set_punctuations = set(string.punctuation)
list_punctuations_out = ['”', '”', "›", "“"]
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)

list_emoji_not_in_lib = [u'\U0001f1fb', u'\ufe0f', u'\u20e3', u'\U0001f3fb', u'\U0001f3fb0', u'\u200e']
list_emoji_keep = ["=(", "=)", ":v", ":3", ":)", ":(", ":D", "^^", "<URL>"]

dict_typing = get_dict_typing_error("/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/dataset/support_data/typing_error_telex.csv")


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


def norm_emoji_make_by_punctuation(text):
    for i in range(3):
        text = text.replace(":))))", ":))")
        text = text.replace(":)))", ":))")
        text = text.replace("=))))", "=))")
        text = text.replace("=)))", "=))")
        text = text.replace(":(((", ":((")
        text = text.replace(":vv", ":v")
        text = text.replace(": \) \) \)", ": \) \)")
        text = text.replace(": \) \)", ": \)")
        text = text.replace("= \) \) \)", "= \) \)")
        text = text.replace("= \) \)", "= \)")
        text = text.replace(": \( \( \(", ": \( \(")
        text = text.replace(": \( \(", ": \(")
        text = text.replace("= \( \( \(", "= \( \(")
        text = text.replace("= \( \(", "= \(")

    text = text.replace(":))", ":)")
    text = text.replace("=))", "=)")
    text = text.replace(":((", ":(")
    text = text.replace("=((", "=(")
    text = text.replace(": \)", ":)")
    text = text.replace("= \)", "=)")
    text = text.replace(": \(", ":(")
    text = text.replace("= \(", "=(")
    text = text.replace(":vv", ":v")
    text = text.replace("\ ?", "?")
    text = text.replace("\\?", "?")

    return text


def handle_emoji(text):
    for e_emoji_punc in list_emoji_keep:
        text = text.replace(e_emoji_punc, " {} ".format(e_emoji_punc))

    list_new_char = []
    for e_char in text:
        if e_char in emoji.UNICODE_EMOJI or e_char in list_emoji_not_in_lib:
            list_new_char.append(" {} ".format(e_char))
        else:
            list_new_char.append(e_char)

    text = "".join(list_new_char)

    return text


def handle_punctuation_one_word(text):
    # need replace | for split field in csv file

    l_new_char = []
    for e_char in text:

        if e_char not in list(set_punctuations):
            l_new_char.append(e_char)
        else:
            l_new_char.append(" {} ".format(e_char))
    text = "".join(l_new_char)

    return text


def handle_punctuation_sent(text):
    text = remove_multi_space(text)
    arr_text = text.split(" ")
    print(arr_text)
    l_new_token = []
    for e_token in arr_text:
        if e_token not in list_emoji_keep:
            e_token = handle_punctuation_one_word(e_token)
        l_new_token.append(e_token)

    return " ".join(l_new_token)


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
    text = handle_punctuation_one_word(text)
    text = fix_typing_error(text, dict_typing)
    text = remove_multi_space(text)
    return text


def handle_text_hate_speech(text, is_lower=False):
    text = normalize_text(text)
    text = fix_typing_error(text, dict_typing)
    text = norm_emoji_make_by_punctuation(text)
    text = handle_emoji(text)
    text = handle_punctuation_sent(text)
    text = remove_multi_space(text)
    if is_lower:
        text = text.lower()
    return text


if __name__ == '__main__':
    '''
    text_test = "hôm nay cos ver không tố lăms"
    dict_typing = get_dict_typing_error("../module_dataset/dataset/support_data/typing_error_telex.csv")
    print(fix_typing_error(text_test, dict_typing))
    '''
    print(handle_text_hate_speech("nhìn con ma nunwgs vl"))
