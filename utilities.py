import json
import glob


def get_dict_typing_error(path_file_dict):
    dict_typing_error = {}
    with open(path_file_dict, 'r') as rf:
        for e_line in rf.readlines():
            arr = e_line.replace("\n", "").split()
            dict_typing_error["{} ".format(arr[1])] = arr[0]

    return dict_typing_error


def get_list_from_file(path_word_list):
    list_word = []
    with open(path_word_list, "r") as rf:
        for e_line in rf.readlines():
            list_word.append(e_line.replace('\n', ''))
    return list_word


def get_dict_synonym(path_file_dict):
    with open(path_file_dict, "r") as rf:
        dict_synonym = json.load(rf)
    return dict_synonym


def load_config(path_file_config):
    with open(path_file_config, "r") as rf:
        cf = json.load(rf)
    return cf


def get_all_path_file_in_folder(path_folder):
    list_path_file = []
    path_folder_recursive = path_folder + "/**"
    for e_file in glob.glob(path_folder_recursive):
        list_path_file.append(e_file)
    return list_path_file


def get_name_folder_file(path_file):
    arr_path = path_file.split("/")
    path_folder = "/".join(arr_path[:-1])
    name_file = arr_path[-1]
    return path_folder, name_file