# # from collections import defaultdict
# #
# # path_file = "/home/trangtv/Documents/project/NewsClassify/dataset/sample_dataset/aapd_train.tsv"
# #
# #
# # dict_full = defaultdict(lambda :0)
# # with open(path_file, "r") as rf:
# #     for e_line in rf.readlines():
# #         label = e_line.split("\t")[0]
# #         for idx, e_label in enumerate(label):
# #             if e_label == '1':
# #                 dict_full[idx] += 1
# #
# # print(dict_full)
# # import time
# # from textblob import TextBlob
# #
# # # time.sleep(2)
# # blob = TextBlob("hôm nay vui đấy")
# # str_en_translate = blob.translate(from_lang='vi', to='en')
# #
# # n_blob = TextBlob(str(str_en_translate))
# # str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')
#
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # tfidf = TfidfVectorizer(min_df=1)
# #
# # corpus = [
# #     'I would like to check this document',
# #     'How about one more document like you',
# #     'Aim is to capture the key words from the corpus',
# #     'frequency of words in a document is called term frequency'
# # ]
# #
# # vect = tfidf.fit_transform(corpus)
# # print(vect[3])
# # a = "hom nay di hoc"
# # for c in a:
# #     print(c)
#
# # print(tfidf.get_feature_names())
# # print(tfidf.idf_)
# # a = [7,3,5,6]
# # list_indicies = sorted(range(len(a)), key=lambda k: a[k])
# # list_indicies.reverse()
# # print(list_indicies)
#
# import torch.nn as nn
# import torch
# #
# a = torch.rand((3, 4, 230))
# lstm_t = nn.LSTM(230, 5, num_layers=1, batch_first=True, bidirectional=True)
# output, (last_hidden, last_cell) = lstm_t(a)
# print(output.shape)
# #
# # # print(last_hidden)
# # # output = output.permute(0, 2, 1)
# # print(output)
# # print(output.shape)
# # print(output[:, -1, :].shape)
# # merged_state = torch.cat([s for s in last_hidden], 1)
# # print(merged_state.shape)
# # torch.Size([2, 10])
# # print(merged_state.shape)
# # merged_state = merged_state.unsqueeze(-1)
# # torch.Size([2, 10, 1])
# # print(merged_state.shape)
# # print(last_hidden.shape)
# # print(last_hidden)
# #
# # print(output.shape)
# #
# # # print(output[:-1:])
# # print(output)
#
# # a = torch.Tensor([[[ 8.9013e-03, -2.1320e-01,  2.9509e-03, -6.5920e-02,  8.4693e-03],
# #          [ 2.0403e-02,  1.2472e-01,  9.8842e-04, -2.0926e-02,  1.4485e-02],
# #          [ 9.1411e-03,  8.8196e-02, -2.0368e-04,  7.4281e-02, -3.1024e-01]],
# #
# #         [[ 1.9224e-01,  7.6485e-01, -7.3278e-04,  6.7092e-01, -8.5206e-02],
# #          [-6.3303e-02,  4.0244e-01, -1.6050e-03,  4.5860e-01, -3.2597e-02],
# #          [-4.5168e-02,  5.3618e-01, -4.2020e-03, -5.5704e-01, -1.4953e-01]]])
# # a = a.permute(1, 0, 2)
# # a = a.reshape(3, 10)
# # print(a.shape)
# # print(a)
#
# # a = last_hidden.permute(1, 0, 2)
#
# # a = a.view((3, 10))
# # print(a.shape)
# # print(a)
# # a = a.reshape(3, -1)
# # print(a)
# # import pandas as pd
# #
# # path_label = "/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/dataset/raw_data/03_train_label.csv"
# # df = pd.read_csv(path_label, sep=",", names=["id", "label"])
# #
# # # for key, value in df.items():
# # #     print(key)
# # #     print(value)
# #
# # # print(dict_id_label['train_dudzhmrivs'])
# # print(df.id.tolist())
# # from module_dataset.preprocess_data.handle_text import *
# # a = "Buồn vl d:((("
# # print(a)
# # b = handle_text_hate_speech(a, is_lower=True)
# # print(b)
#
# # a = ["1", "2", '3']
# # print(a)
# # a.remove("1")
# # print(a)
# #
# #
# #
# # a = "trang la ta day nhe"
# # print(a.find("la ta"))
# # from sklearn.model_selection import StratifiedKFold
# #
# # a = ["trang", "la", "ta", "day","trang", "la", "ta", "day", "ta"]
# # b = [1,2,1,1,2, 2,1,2, 2]
# # kf = StratifiedKFold(n_splits=5, shuffle=True)
# # print(kf.get_n_splits(a, b))
# # for train_index, test_index in kf.split(a, b):
# #     print("start print")
# #     print(train_index)
# #     print(test_index)
# #     print("end print")
# # from module_dataset.preprocess_data.handle_text import *
# # a = "   trang la     ta day   . . dada "
# # print(remove_multi_space(a))
# # a = "hom nnay phai  la trang la ta  dauy"
# # b = "la"
# # print(a.find(b))
# # from module_dataset.preprocess_data.handle_data_augmentation import *
# # print(get_number_token_with_length(a, 15))
# # print(a.split(" "))
# from collections import defaultdict
# # a = "1, 2,3"
# # b = "4"
# # print(a + b)
# # a = [1,2,0,1,2,1]
# #
# # b = [1,2,0,1,1,2]
# # # list_true = [0] * 3
# # # list_total_each_value = [0] * 3
# # # for idx, value_b in enumerate(b):
# # #     list_total_each_value[a[idx]] += 1
# # #     if value_b == a[idx]:
# # #         list_true[value_b] += 1
# # #
# # # print(list_true)
# # # print(list_total_each_value)
# #
# # list_true = [0,3,4]
# # list_total = [2,5,6]
# # a = "_".join(str(x) for x in list_total)
# # print(a)
# # print(l_new_total))
# # from operator import add
# # # print(list_true + list_total)
# # list_true = list(map(add, list_true, list_total))
# # print(list_true)
#
# from module_dataset.preprocess_data.handle_data_augmentation import *
# # a = "Thưởng nóng cho anh em nào biết con chó này . Địt cái lồn mẹ nhà mày mày bảo đăng bậy thì ra đây gặp bố . Bố nhét cứt vào mồm mày luôn . <URL>"
# # ab = back_translate_data(a)
# # print(type(ab))
# # print(ab)
# # print(str(ab))
# # print(type(str(ab)))
# # import string
# # a = list(string.ascii_lowercase)
# # shuffle(a)
# # print(a[0])
# # print(random_remove_insert_character("trang", thresh_hold_active_char=0.5))
# #
# # l_a = ["mai", "hung", "trang"]
# # dict_sym = {}
# # for e_token in l_a:
# #     n_tmp = l_a.copy()
# #     print(n_tmp)
# #     n_tmp.remove(e_token)
# #     dict_sym[e_token] = n_tmp
# #
# # print(dict_sym)
#
# from utilities import *
# # print(get_all_path_file_in_folder("/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/dataset/data_for_train/dl/data_k_fold_not_augment/train_dl_id"))
#
# # train_f1 = [0] * 2000 + [1] * 10000 + [2] * 8000
# # test_f1 = [0] * 3000 + [1] * 7000 + [2] * 10000
# # from sklearn.metrics import f1_score
# # a = f1_score(train_f1, test_f1, average=None)
# # print(a.tolist())
# # print(type(a.tolist()))
# # import pickle
# # w2v_dict = pickle.load(open("/home/trangtv/Documents/project/HateSpeechDectection/module_dataset/dataset/support_data/dict_map_comment.pkl", 'rb'))
# # print(w2v_dict['comment']['mỏi'])
# # a = {"trang": "la"}
# # a.update({"trang":"b"})
# # print(a)
# # import numpy as np
# # a = [[8.266836166381836, -3.480959415435791, -4.336638450622559]]
# # b = [0.8]
# # n_a = []
# # for idx, e_a in enumerate(a):
# #     n_a.append([float(i) * b[idx] for i in e_a])
# #
# # print(n_a)
# #
# #
# import torch
# import torch.nn.functional as F
# a = torch.Tensor([1,2,-3])
# print(a)
# d = F.log_softmax(a, dim=-1)
# print(d)
# F.dropout2d(a, 0.2, )
d = {"a": [2 ]}
for i in range(3):
    e = d.copy()
    e.update({"a": 4})
    print(e)
    print(d)



# dict_model = {
#     "model_1": {
#         "type_model": "lstm_cnn_word_case_1",
#         "folder_model": "a",
#         "list_checkpoint": [],
#         "list_weighted": [1, 1]
#     },
#     "model_2": {
#         "type_model": "lstm_cnn_word_case_1",
#         "folder_model": "b",
#         "list_checkpoint": [],
#         "list_weighted": [1, 1]
#     }
# }
# for e_key, e_value in dict_model.items():
#     print(e_key)
# # print(dict_model)