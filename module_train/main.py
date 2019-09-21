from module_train.model_architecture_dl.cnn_classify import CNNClassifyWordCharNgram
from module_train.model_architecture_dl.lstm_cnn_word_char_based import LSTMCNNWordCharBase
from module_train.model_architecture_dl.lstm_cnn_word_char_lm import LSTMCNNWordCharLM
from module_dataset.preprocess_data.hanlde_dataloader import *

from module_train.train_dl import Trainer
from module_train.train_ml import *

from module_evaluate import reference_model
from module_evaluate import evaluate

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
                                      batch_size=cf_common['batch_size'])

        data_train_iter = data['iters'][0]

        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = CNNClassifyWordCharNgram.create(cf_common['path_save_model']+cf_common['type_model'],
                                                cf_model,
                                                data['vocabs'])

    elif type_model == "lstm_cnn_lm":
        data = load_data_lm_word_char(path_data,
                                      path_data_train,
                                      path_data_test,
                                      min_freq_word=cf_common['min_freq_word'],
                                      min_freq_char=cf_common['min_freq_char'],
                                      batch_size=cf_common['batch_size'])

        data_train_iter = data['iters'][0]
        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = LSTMCNNWordCharLM.create(cf_common['path_save_model']+cf_common['type_model'],
                                         cf_model,
                                         data['vocabs'])

    elif type_model == "lstm_cnn_word_char_base":
        data = load_data_word_lstm_char(path_data,
                                        path_data_train,
                                        path_data_test,
                                        min_freq_word=cf_common['min_freq_word'],
                                        min_freq_char=cf_common['min_freq_char'],
                                        batch_size=cf_common['batch_size'])

        data_train_iter = data['iters'][0]
        if path_data_test is not None:
            data_test_iter = data['iters'][1]
        else:
            data_test_iter = None

        model = LSTMCNNWordCharBase.create(cf_common['path_save_model']+cf_common['type_model'],
                                           cf_model,
                                           data['vocabs'])

    trainer = Trainer(cf_common['path_save_model']+cf_common['type_model'],
                      model,
                      cf_model,
                      cf_common['prefix_model'],
                      cf_common['log_file'],
                      data_train_iter,
                      data_test_iter)

    trainer.train(cf_common['num_epochs'])


def train_ml(path_save_model, path_data, name_train, name_test=None):
    result = train_all_ml_model(path_save_model, path_data, name_train, name_test)
    print(result)


if __name__ == '__main__':
    cf_common = {
        "path_save_model": "save_model/",
        "path_data": "../module_dataset/dataset/data_for_train/",
        "path_data_train": "exp_train.csv",
        "path_data_test": "exp_test.csv",
        "prefix_model": "model_lm_lstm_cnn",
        "log_file": "log_file_13.txt",
        "type_model": "cnn_classify",
        "num_epochs": 50,
        "min_freq_word": 2,
        "min_freq_char": 2,
        "batch_size": 3
    }

    cf_model_cnn_classify = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 64,
        'filter_num': 15,
        'kernel_sizes': [2, 3, 4],
        'dropout_cnn': 0.5,
        'dropout_ffw': 0.5,
        'learning_rate': 0.001,
        'weight_decay': 0
    }

    cf_model_char_base = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 64,
        'hidden_size_word': 32,
        'hidden_size_char_lstm': 32,
        'use_highway_char': False,
        'use_char_cnn': False,
        'use_last_as_ft': True,
        'char_cnn_filter_num': 15,
        'char_window_size': [2, 3],
        'dropout_cnn': 0.5,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'weight_decay': 0
    }

    cf_model_lstm_cnn_lm = {
        'use_xavier_weight_init': True,
        'word_embedding_dim': 200,
        'char_embedding_dim': 64,
        'hidden_size_word': 5,
        'use_highway_char': False,
        'use_last_as_ft': True,
        'hidden_size_reduce': 256,
        'use_char_cnn': True,
        'char_cnn_filter_num': 15,
        'char_window_size': [2, 3],
        'dropout_cnn': 0.5,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'weight_decay': 10
    }

    if cf_common['type_model'] == 'cnn_classify':
        train_model_dl(cf_common, cf_model_cnn_classify)

    elif cf_common['type_model'] == 'lstm_cnn_word_char_base':
        train_model_dl(cf_common, cf_model_char_base)

    else:
        train_model_dl(cf_common, cf_model_lstm_cnn_lm)










