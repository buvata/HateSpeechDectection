from module_train.train_ml import *
from module_train.model_architecture_dl.cnn_classify import *
from module_train.model_architecture_dl.lstm_cnn_word_char_based import *
from module_train.model_architecture_dl.lstm_cnn_word_char_lm import *
from module_train.model_architecture_dl.lstm_cnn_word import *
import pickle
import numpy as np

from torchtext import data
from utilities import *


def get_input_processor_words(inputs, type_model, vocab_word, vocab_char=None):
    if type_model == "word_char_based" or type_model == "lstm_cnn_word_case_1":

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

    elif type_model == "lstm_cnn_word_case_1":
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

            elif type_model == "lstm_cnn_word_case_1":
                model = LSTMCNNWord.load(path_save_model, e_checkpoint)

            vocab_word, vocab_char, vocab_label = model.vocabs
            print(vocab_word.stoi)
            print(vocab_char.stoi)
            print(vocab_label.stoi)
            for idx, e_sent in enumerate(list_test_sent):

                data_e_line = get_input_processor_words(e_sent, type_model, vocab_word, vocab_char)
                predict = model(data_e_line)

                print("{}|{}\n".format(e_sent, torch.max(predict, 1)[1].numpy().tolist()[0]))

                e_predict = predict.numpy().tolist()[0]
                dict_predict[list_id[idx]].append(e_predict)

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

    path_save_model_1 = "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_2"
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
    list_path_checkpoint = ["/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify/model_cnn_classify_fold_1_epoch_23_train_loss_0.4012_macro0.7323_full__0.93_0.68_0.58_test_loss_0.2128_macro0.7072_full__0.98_0.52_0.63",
                         "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify/model_cnn_classify_fold_2_epoch_25_train_loss_0.3752_macro0.754_full__0.93_0.7_0.63_test_loss_0.2436_macro0.6709_full__0.97_0.5_0.54",
                         "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify/model_cnn_classify_fold_3_epoch_11_train_loss_0.5021_macro0.6655_full__0.9_0.58_0.51_test_loss_0.2644_macro0.671_full__0.97_0.45_0.59",
                         "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify/model_cnn_classify_fold_4_epoch_21_train_loss_0.3905_macro0.7413_full__0.93_0.7_0.58_test_loss_0.2131_macro0.6711_full__0.97_0.48_0.56",
                         "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify/model_cnn_classify_fold_7_epoch_18_train_loss_0.4213_macro0.7123_full__0.92_0.65_0.56_test_loss_0.2242_macro0.6802_full__0.97_0.49_0.58"]

    dict_model = {
        "model_1": {
            "type_model": "lstm_cnn_word_case_1",
            "folder_model": path_save_model_1,
            "list_checkpoint": ["/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_2/case_4_cnn_char_hidden_word_64_ws_2_8_spatial_drop__epoch_16_train_loss_0.2536_macro0.8535_full__0.96_0.79_0.81",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_2/case_4_cnn_char_hidden_word_64_ws_2_8_spatial_drop__epoch_21_train_loss_0.2131_macro0.882_full__0.96_0.82_0.86",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_2/case_4_cnn_char_hidden_word_64_ws_2_8_spatial_drop__epoch_27_train_loss_0.1779_macro0.9022_full__0.96_0.85_0.89",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_2/case_4_cnn_char_hidden_word_64_ws_2_8_spatial_drop__epoch_30_train_loss_0.1605_macro0.9117_full__0.97_0.86_0.91"
                                ],
            "list_weighted": [1, 1, 1, 1]
        },
        "model_2": {
            "type_model": "lstm_cnn_word_case_1",
            "folder_model": path_save_model_2,
            "list_checkpoint": ["/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_3/case_6_lstm_char_hidden_word_64_ws_1_16_spatial_drop__epoch_15_train_loss_0.2865_macro0.8294_full__0.95_0.77_0.77",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_3/case_6_lstm_char_hidden_word_64_ws_1_16_spatial_drop__epoch_20_train_loss_0.2334_macro0.8681_full__0.96_0.83_0.81",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_3/case_6_lstm_char_hidden_word_64_ws_1_16_spatial_drop__epoch_26_train_loss_0.1943_macro0.891_full__0.96_0.86_0.86",
                                "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/lstm_cnn_word_case_3/case_6_lstm_char_hidden_word_64_ws_1_16_spatial_drop__epoch_30_train_loss_0.1701_macro0.9041_full__0.97_0.88_0.87"],
            "list_weighted": [1, 1, 1, 1]
        }
    }

    get_average_predict_model("submit.txt", path_data_test,
                              dict_model)
