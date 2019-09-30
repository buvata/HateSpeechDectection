# import sys
# sys.path.append("/data/trangtv_workspace/HateSpeechDectection")
from module_train.model_architecture_dl.cnn_classify import CNNClassifyWordCharNgram
from module_train.model_architecture_dl.lstm_cnn_word_char_based import LSTMCNNWordCharBase
from module_train.model_architecture_dl.lstm_cnn_word_char_lm import LSTMCNNWordCharLM
from module_dataset.preprocess_data.hanlde_dataloader import *

from module_train.train_dl import Trainer
from module_train.train_ml import *


def train_model_dl(cf_common, cf_model):
    path_data = cf_common['path_data']
    path_data_train = cf_common['path_data_train']
    path_data_test = cf_common['path_data_test']

    type_model = cf_common['type_model']
    model = None
    data_train_iter = None
    data_test_iter = None

    if type_model == "cnn_classify":
        data = load_data_lm_word_char(path_data,
                                      path_data_train,
                                      path_data_test,
                                      min_freq_word=cf_common['min_freq_word'],
                                      min_freq_char=cf_common['min_freq_char'],
                                      batch_size=cf_common['batch_size'],
                                      cache_folder=cf_common['cache_folder'],
                                      name_vocab=cf_common['name_vocab'],
                                      path_vocab_pre_built=cf_common['path_vocab_pre_built'])

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = CNNClassifyWordCharNgram.create(cf_common['path_save_model'] + cf_common['type_model'],
                                                cf_model,
                                                data['vocabs'])

    elif type_model == "lstm_cnn_lm":
        data = load_data_lm_word_char(path_data,
                                      path_data_train,
                                      path_data_test,
                                      min_freq_word=cf_common['min_freq_word'],
                                      min_freq_char=cf_common['min_freq_char'],
                                      batch_size=cf_common['batch_size'],
                                      cache_folder=cf_common['cache_folder'],
                                      name_vocab=cf_common['name_vocab'],
                                      path_vocab_pre_built=cf_common['path_vocab_pre_built'])

        data_train_iter = data['iters'][0]
        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = LSTMCNNWordCharLM.create(cf_common['path_save_model'] + cf_common['type_model'],
                                         cf_model,
                                         data['vocabs'])

    elif type_model == "lstm_cnn_word_char_base":
        data = load_data_word_lstm_char(path_data,
                                        path_data_train,
                                        path_data_test,
                                        min_freq_word=cf_common['min_freq_word'],
                                        min_freq_char=cf_common['min_freq_char'],
                                        batch_size=cf_common['batch_size'],
                                        cache_folder=cf_common['cache_folder'],
                                        path_vocab_pre_built=cf_common['path_vocab_pre_built'],)

        data_train_iter = data['iters'][0]
        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = LSTMCNNWordCharBase.create(cf_common['path_save_model'] + cf_common['type_model'],
                                           cf_model,
                                           data['vocabs'])

    print("!!Load dataset done !!\n")
    trainer = Trainer(cf_common['path_save_model']+cf_common['type_model'],
                      model,
                      cf_model,
                      cf_common['prefix_model'],
                      cf_common['log_file'],
                      len(data['vocabs'][2]),
                      data_train_iter,
                      data_test_iter)

    trainer.train(cf_common['num_epochs'])


def train_ml(path_save_model, path_data, name_train, name_test=None):
    result = train_all_ml_model(path_save_model, path_data, name_train, name_test)
    print(result)


if __name__ == '__main__':
    cf_common = {
        "path_save_model": "save_model/",
        "path_data": "../module_dataset/dataset/data_for_train/dl/data_k_fold/",
        "path_data_train": "train_dl_1",
        "path_data_test": "validation_dl_1",
        "prefix_model": "cnn_fold_1",
        "log_file": "log_fold_1_cnn.txt",
        "type_model": "lstm_cnn_lm",
        "num_epochs": 25,
        "min_freq_word": 2,
        "min_freq_char": 5,
        "path_vocab_pre_built": "save_model/vocabs_all.pt",
        "cache_folder": "../module_dataset/dataset/support_data",
        "name_vocab": "out_embedding.txt",
        "batch_size": 32
    }

    cf_model_cnn_classify = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 32,
        'filter_num_word': 12,
        'filter_num_char': 10,
        'kernel_sizes_word': [2, 3, 4],
        'kernel_sizes_char': [2, 3],
        'dropout_cnn': 0.6,
        'dropout_ffw': 0.55,
        'learning_rate': 0.0002,
        'weight_decay': 0,
        'D_cnn': "1_D"
    }

    cf_model_char_base = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 64,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 32,
        'use_highway_char': False,
        'use_char_cnn': True,
        'D_cnn': '1_D',
        'use_last_as_ft': True,
        'char_cnn_filter_num': 5,
        'char_window_size': [2, 3],
        'dropout_cnn': 0.5,
        'dropout_rate': 0.5,
        'learning_rate': 0.0005,
        'weight_decay': 0
    }

    cf_model_lstm_cnn_lm = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 64,
        'hidden_size_word': 32,
        'use_highway_char': False,
        'use_last_as_ft': True,
        'hidden_size_reduce': 128,
        'use_char_cnn': True,
        'D_cnn': '2_D',
        'char_cnn_filter_num': 10,
        'char_window_size': [2, 3],
        'dropout_cnn': 0.55,
        'dropout_rate': 0.55,
        'learning_rate': 0.0005,
        'weight_decay': 0
    }

    for j in range(3):
        if j == 0:
            cf_common.update({"type_model": "cnn_classify"})
        if j == 1:
            cf_common.update({"type_model": "lstm_cnn_word_char_base"})
        if j == 2:
            cf_common.update({"type_model": "lstm_cnn_lm"})

        for i in range(1, 9):
            path_data_train = "train_dl_id_{}_augment".format(i)
            path_data_test = "validation_dl_{}".format(i)
            cf_common.update({"path_data_train": path_data_train})
            cf_common.update({"path_data_test": path_data_test})

            cf_common.update({"prefix_model": "model_{}_fold_{}".format(cf_common['type_model'], i)})
            cf_common.update({"log_file": "log_{}_fold_{}.txt".format(cf_common['type_model'], i)})

            if cf_common['type_model'] == 'cnn_classify':
                train_model_dl(cf_common, cf_model_cnn_classify)

            elif cf_common['type_model'] == 'lstm_cnn_word_char_base':
                train_model_dl(cf_common, cf_model_char_base)

            elif cf_common['type_model'] == "lstm_cnn_lm":
                train_model_dl(cf_common, cf_model_lstm_cnn_lm)
