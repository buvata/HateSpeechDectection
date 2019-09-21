from module_train.train_ml import *
from module_train.model_architecture_dl.cnn_classify import *
from module_train.model_architecture_dl.lstm_cnn_word_char_based import *
from module_train.model_architecture_dl.lstm_cnn_word_char_lm import *

from torchtext import data
from utilities import *


def get_input_processor_words(inputs, type_model, vocab_word, vocab_char=None):
    if type_model == "word_char_based":

        inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char = data.NestedField(inputs_char_nesting,
                                       init_token="<bos>", eos_token="<eos>")

        inputs_word.vocab = vocab_word
        if vocab_char is not None:
            inputs_char.vocab = vocab_char
            fields = [(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))]
        else:
            fields = [('inputs_word', inputs_word)]

        if not isinstance(inputs, list):
            inputs = [inputs]

        examples = []

        for line in inputs:
            examples.append(data.Example.fromlist([line], fields))

        dataset = data.Dataset(examples, fields)
        batchs = data.Batch(data=dataset,
                            dataset=dataset,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    else:
        tokenize_word = lambda x: x.split()
        inputs_word = data.Field(tokenize=tokenize_word, init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_word.vocab = vocab_word
        if vocab_char is not None:
            inputs_char.vocab = vocab_char
            fields = [(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))]
        else:
            fields = [('inputs_word', inputs_word)]

        if not isinstance(inputs, list):
            inputs = [inputs]

        examples = []

        for line in inputs:
            examples.append(data.Example.fromlist([line], fields))

        dataset = data.Dataset(examples, fields)
        batchs = data.Batch(data=dataset,
                            dataset=dataset,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Entire input in one batch
    return batchs


def get_predict_ml(path_data_test, path_save_model):
    x_test = load_data_ml(path_data_test, is_train=False)
    model_ml = load_model_ml(path_save_model)

    y_test = model_ml.predict(x_test[0])

    return y_test


def get_predict_dl(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify'):
    list_test_sent = get_list_from_file(path_data_test)

    if type_model == "cnn_classify":
        model = CNNClassifyWordCharNgram.load(path_save_model, path_model_checkpoint)

    elif type_model == "lstm_cnn_word_char_based":
        model = LSTMCNNWordCharBase.load(path_save_model, path_model_checkpoint)

    else:
        model = LSTMCNNWordCharLM.load(path_save_model, path_model_checkpoint)

    vocab_word, vocab_char, vocab_label = model.vocabs
    # print(vocab_word)

    list_predicts = []
    for e_sent in list_test_sent:
        data_e_line = get_input_processor_words(e_sent, type_model, vocab_word, vocab_char)
        predict = model(data_e_line)
        list_predicts.append(predict)

    return list_predicts

if __name__ == '__main__':
    '''
    path_data_test = "../module_train/save_model/model_ml/data_ft.pkl"
    path_save_model = "../module_train/save_model/model_ml/random_forest_model.pkl"
    result = get_predict_ml(path_data_test, path_save_model)
    print(result)
    '''
    path_save_model = "../module_train/save_model/cnn_classify"
    path_data_test = "../module_dataset/dataset/data_for_train/exp_test.pkl"
    path_model_checkpoint = "../module_train/save_model/cnn_classify/model_lm_lstm_cnn_epoch_0_train_acc_0.6_loss_1.0921_test_acc_1.0_loss_0.3284"
    result = get_predict_dl(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify')

    print(result)

