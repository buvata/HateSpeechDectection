from module_train.train_ml import *
from module_train.model_architecture_dl.cnn_classify import *
from module_train.model_architecture_dl.lstm_cnn_word_char_based import *
from module_train.model_architecture_dl.lstm_cnn_word_char_lm import *
from module_train.model_architecture_dl.lstm_cnn_word import *
import pickle
import numpy as np

from torchtext import data
from utilities import *
import torch.nn.functional as F


def get_input_processor_words(inputs, type_model, vocab_word, vocab_char=None):
    if type_model == "lstm_cnn_word_char_based" or type_model == "lstm_cnn_word":

        inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

        inputs_char = data.NestedField(inputs_char_nesting,
                                       init_token="<bos>", eos_token="<eos>")

        inputs_word.vocab = vocab_word
        if vocab_char is not None:
            inputs_char.vocab = inputs_char_nesting.vocab = vocab_char
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
    list_id, list_test_sent = get_list_test_id_from_file(path_data_test)

    if type_model == "cnn_classify":
        model = CNNClassifyWordCharNgram.load(path_save_model, path_model_checkpoint)

    elif type_model == "lstm_cnn_word_char_based":
        model = LSTMCNNWordCharBase.load(path_save_model, path_model_checkpoint)

    elif type_model == "lstm_cnn_lm":
        model = LSTMCNNWordCharLM.load(path_save_model, path_model_checkpoint)

    elif type_model == "lstm_cnn_word":
        model = LSTMCNNWord.load(path_save_model, path_model_checkpoint)

    vocab_word, vocab_char, vocab_label = model.vocabs
    # print(vocab_word)

    list_predicts = []
    for idx, e_sent in enumerate(list_test_sent):

        data_e_line = get_input_processor_words(e_sent, type_model, vocab_word, vocab_char)
        predict = model(data_e_line)
        print(predict.numpy().tolist()[0])
        # predict_value = torch.max(predict, 1)[1].numpy().tolist()[0]
        # if predict_value != 0:
        #     print("{}|{}".format(list_id[idx], e_sent))
        list_predicts.append(predict)

    return list_predicts


def get_average_predict_model(path_submission, path_data_test,
                              dict_model=None,
                              list_weighted=None):

    list_id, list_test_sent = get_list_test_id_from_file(path_data_test)
    dict_predict = defaultdict(list)

    for e_key, e_value in dict_model.items():
        type_model = e_value['type_model']
        list_checkpoints = e_value['list_checkpoint']
        path_save_model = e_value['folder_model']

        for e_checkpoint in list_checkpoints:
            if type_model == "cnn_classify":
                model = CNNClassifyWordCharNgram.load(path_save_model, e_checkpoint)

            elif type_model == "lstm_cnn_word_char_based":
                model = LSTMCNNWordCharBase.load(path_save_model, e_checkpoint)

            elif type_model == "lstm_cnn_lm":
                model = LSTMCNNWordCharLM.load(path_save_model, e_checkpoint)

            elif type_model == "lstm_cnn_word":
                model = LSTMCNNWord.load(path_save_model, e_checkpoint)

            vocab_word, vocab_char, vocab_label = model.vocabs
            for idx, e_sent in enumerate(list_test_sent):

                data_e_line = get_input_processor_words(e_sent, type_model, vocab_word, vocab_char)
                predict = model(data_e_line)

                print("{}|{}\n".format(e_sent, torch.max(predict, 1)[1].numpy().tolist()[0]))

                e_predict = predict[0]
                output_score_soft_max = F.softmax(e_predict).numpy().tolist()
                dict_predict[list_id[idx]].append(output_score_soft_max)

    with open("dict_predict_cnn.pkl", "wb") as dict_write:
        pickle.dump(dict_predict, dict_write, protocol=pickle.HIGHEST_PROTOCOL)

    with open("dict_predict_cnn.pkl", "rb") as read_dict:
        dict_predict = pickle.load(read_dict)

    with open(path_submission, "w") as wf:
        wf.write("id,label_id\n")

        for e_key, e_predict_list in dict_predict.items():
            if list_weighted is not None:
                n_predict_list = []
                for idx, e_predict in enumerate(e_predict_list):
                    n_predict_list.append([e_list * list_weighted[idx] for e_list in e_predict])
            else:
                n_predict_list = e_predict_list

            np_arr_predict = np.array(n_predict_list)
            final_predict = np.mean(np_arr_predict, 0)
            predict_label = np.argmax(final_predict, 0)
            line_write = "{},{}\n".format(e_key, predict_label)
            wf.write(line_write)


if __name__ == '__main__':

    path_save_model_1 = "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_1"
    path_save_model_2 = "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_3"
    path_data_test = "../module_dataset/dataset/data_for_train/dl/data_full/test_process_emoji_punct.csv"
    path_model_checkpoint = "../module_train/save_model/cnn_classify/" \
                            "model_cnn_classify_fold_1_epoch_23_train_loss_0.4012_macro0.7323_full__0.93_0.68_0.58_test_loss_0.2128_macro0.7072_full__0.98_0.52_0.63"
    # result = get_predict_dl(path_data_test, path_save_model, path_model_checkpoint, type_model='cnn_classify')
    # list_path_file = get_all_path_file_in_folder(path_save_model_1)
    # list_path_checkpoint = []
    # for e_path_file in list_path_file:
    #     if "model_cnn_classify_fold" in e_path_file:
    #         list_path_checkpoint.append(e_path_file)
    dict_model = {
        "model_1": {
            "type_model": "lstm_cnn_word_char_based",
            "folder_model": path_save_model_1,
            "list_checkpoint": ["/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_1/case_1_lstm_lstm_word_char__epoch_11_train_loss_0.1406_macro0.7073_full__0.98_0.52_0.62"
                                ],
            "list_weighted": [1]
        }
    }

    get_average_predict_model("submit.txt", path_data_test,
                              dict_model)
