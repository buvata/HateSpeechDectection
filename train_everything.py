# from collections import defaultdict
#
# path_file = "/home/trangtv/Documents/project/NewsClassify/dataset/sample_dataset/aapd_train.tsv"
#
#
# dict_full = defaultdict(lambda :0)
# with open(path_file, "r") as rf:
#     for e_line in rf.readlines():
#         label = e_line.split("\t")[0]
#         for idx, e_label in enumerate(label):
#             if e_label == '1':
#                 dict_full[idx] += 1
#
# print(dict_full)
# import time
# from textblob import TextBlob
#
# # time.sleep(2)
# blob = TextBlob("hôm nay vui đấy")
# str_en_translate = blob.translate(from_lang='vi', to='en')
#
# n_blob = TextBlob(str(str_en_translate))
# str_vn_back_translate = n_blob.translate(from_lang='en', to='vi')

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(min_df=1)
#
# corpus = [
#     'I would like to check this document',
#     'How about one more document like you',
#     'Aim is to capture the key words from the corpus',
#     'frequency of words in a document is called term frequency'
# ]
#
# vect = tfidf.fit_transform(corpus)
# print(vect[3])
# a = "hom nay di hoc"
# for c in a:
#     print(c)

# print(tfidf.get_feature_names())
# print(tfidf.idf_)
# a = [7,3,5,6]
# list_indicies = sorted(range(len(a)), key=lambda k: a[k])
# list_indicies.reverse()
# print(list_indicies)

import torch.nn as nn
import torch

a = torch.rand((3, 4, 230))
lstm_t = nn.LSTM(230, 5, num_layers=1, batch_first=True, bidirectional=True)
output, (last_hidden, last_cell) = lstm_t(a)

# print(last_hidden)
# output = output.permute(0, 2, 1)
print(output)
print(output.shape)
print(output[:, -1, :].shape)
# merged_state = torch.cat([s for s in last_hidden], 1)
# print(merged_state.shape)
# torch.Size([2, 10])
# print(merged_state.shape)
# merged_state = merged_state.unsqueeze(-1)
# torch.Size([2, 10, 1])
# print(merged_state.shape)
# print(last_hidden.shape)
# print(last_hidden)
#
# print(output.shape)
#
# # print(output[:-1:])
# print(output)

# a = torch.Tensor([[[ 8.9013e-03, -2.1320e-01,  2.9509e-03, -6.5920e-02,  8.4693e-03],
#          [ 2.0403e-02,  1.2472e-01,  9.8842e-04, -2.0926e-02,  1.4485e-02],
#          [ 9.1411e-03,  8.8196e-02, -2.0368e-04,  7.4281e-02, -3.1024e-01]],
#
#         [[ 1.9224e-01,  7.6485e-01, -7.3278e-04,  6.7092e-01, -8.5206e-02],
#          [-6.3303e-02,  4.0244e-01, -1.6050e-03,  4.5860e-01, -3.2597e-02],
#          [-4.5168e-02,  5.3618e-01, -4.2020e-03, -5.5704e-01, -1.4953e-01]]])
# a = a.permute(1, 0, 2)
# a = a.reshape(3, 10)
# print(a.shape)
# print(a)

# a = last_hidden.permute(1, 0, 2)

# a = a.view((3, 10))
# print(a.shape)
# print(a)
# a = a.reshape(3, -1)
# print(a)
