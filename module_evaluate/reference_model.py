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
                              dict_model=None):

    list_id, list_test_sent = get_list_test_id_from_file(path_data_test)
    dict_predict = defaultdict(list)
    dict_predict_raw = defaultdict(list)

    for e_key, e_value in dict_model.items():
        type_model = e_value['type_model']
        list_checkpoints = e_value['list_checkpoint']
        path_save_model = e_value['folder_model']

        for e_checkpoint in list_checkpoints:
            arr_check = e_checkpoint.split("macro")
            fold_1 = float(arr_check[1].split("_")[0])
            fold_2 = float(arr_check[2].split("_")[0])
            weighted_checkpoint = (fold_1 + fold_2 * 2) / 3

            e_checkpoint = os.path.join(path_save_model, e_checkpoint)
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

                # print("{}|{}\n".format(e_sent, torch.max(predict, 1)[1].cpu().numpy().tolist()[0]))

                e_predict = predict[0]
                output_score_soft_max = F.softmax(e_predict).cpu().numpy().tolist()
                dict_predict_raw[list_id[idx]].append(output_score_soft_max)
                # output_score_soft_max = [e_score * weighted_checkpoint for e_score in output_score_soft_max]
                predict_point = torch.max(predict, 1)[1].cpu().numpy().tolist()[0]

                dict_predict[list_id[idx]].append(predict_point)

    with open("dict_predict_cnn.pkl", "wb") as dict_write:
        pickle.dump(dict_predict, dict_write, protocol=pickle.HIGHEST_PROTOCOL)

    with open("dict_predict_raw.pkl", "wb") as dict_write_raw:
        pickle.dump(dict_predict_raw, dict_write_raw, protocol=pickle.HIGHEST_PROTOCOL)

    with open("dict_predict_cnn.pkl", "rb") as read_dict:
        dict_predict = pickle.load(read_dict)

    with open("dict_predict_raw.pkl", "rb") as dict_read_raw:
        dict_predict_raw = pickle.load(dict_read_raw)

    with open(path_submission, "w") as wf:
        wf.write("id,label_id\n")

        for e_key, e_predict_list in dict_predict.items():
            # if list_weighted is not None:
            #     n_predict_list = []
            #     for idx, e_predict in enumerate(e_predict_list):
            #         n_predict_list.append([e_list * list_weighted[idx] for e_list in e_predict])
            # else:
            #     n_predict_list = e_predict_list

            # np_arr_predict = np.array(e_predict_list)
            # final_predict = np.mean(np_arr_predict, 0)
            # predict_label = np.argmax(final_predict, 0)
            print(e_key, e_predict_list)
            if e_predict_list.count(1) == 1 and e_predict_list.count(2) == 0 or \
                    e_predict_list.count(2) == 1 and e_predict_list.count(1) == 0:
                print("check___")
                print(dict_predict_raw[e_key])
                print("check___")
            predict_label = check_list_vote(e_predict_list)
            line_write = "{},{}\n".format(e_key, predict_label)
            wf.write(line_write)


def check_list_vote(list_predict):
    count_predict_not_zero = 0
    l_predict_not_zero = []
    for e_predict in list_predict:
        if e_predict != 0:
            count_predict_not_zero += 1
            l_predict_not_zero.append(e_predict)

    if count_predict_not_zero == 1:
        predict_label = l_predict_not_zero[0]
    if count_predict_not_zero == 3 or count_predict_not_zero ==5 or count_predict_not_zero == 7:
        predict_label = max(set(l_predict_not_zero), key=l_predict_not_zero.count)
    if count_predict_not_zero == 0 or count_predict_not_zero == 2:
        predict_label = 0
    if count_predict_not_zero == 4 or count_predict_not_zero == 6 or count_predict_not_zero == 8:
        if l_predict_not_zero.count(2) == l_predict_not_zero.count(1):
            predict_label = 1
        elif l_predict_not_zero.count(2) > l_predict_not_zero.count(1):
            predict_label = 2
        else:
            predict_label = 1
    return predict_label





if __name__ == '__main__':

    path_save_model_1 = "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/cnn_classify_case_1"
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
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_1",
            "list_checkpoint": ["model_cnn_classify_fold_1_epoch_30_train_loss_0.1542_macro0.7232_full__0.98_0.6_0.59_test_loss_0.1922_macro0.7129_full__0.98_0.49_0.68",
                                # "model_cnn_classify_fold_1_epoch_31_train_loss_0.1485_macro0.713_full__0.98_0.58_0.58_test_loss_0.1913_macro0.702_full__0.98_0.48_0.65",
                                ],
        },
        "model_2": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_2",
            "list_checkpoint": [
                "model_cnn_classify_fold_2_epoch_29_train_loss_0.1591_macro0.708_full__0.98_0.56_0.59_test_loss_0.1718_macro0.6987_full__0.98_0.56_0.56",
                # "model_cnn_classify_fold_2_epoch_31_train_loss_0.1501_macro0.7278_full__0.98_0.58_0.62_test_loss_0.1719_macro0.7019_full__0.98_0.55_0.58",
                ],
        },
        "model_3": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_3",
            "list_checkpoint": [
                # "model_cnn_classify_fold_3_epoch_31_train_loss_0.1548_macro0.7019_full__0.98_0.54_0.58_test_loss_0.198_macro0.67_full__0.98_0.48_0.55",
                "model_cnn_classify_fold_3_epoch_32_train_loss_0.151_macro0.7073_full__0.98_0.54_0.6_test_loss_0.1986_macro0.6776_full__0.98_0.47_0.58"
                ],
        },
        "model_4": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_4",
            "list_checkpoint": [
                "model_cnn_classify_fold_4_epoch_42_train_loss_0.1575_macro0.6965_full__0.98_0.53_0.58_test_loss_0.1909_macro0.6916_full__0.98_0.52_0.58",
                # "model_cnn_classify_fold_4_epoch_48_train_loss_0.1461_macro0.7293_full__0.98_0.6_0.61_test_loss_0.1976_macro0.692_full__0.97_0.51_0.59"
                ],
        },
        "model_5": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_5",
            "list_checkpoint": [
                "model_cnn_classify_fold_5_epoch_33_train_loss_0.1453_macro0.7054_full__0.98_0.55_0.59_test_loss_0.1799_macro0.7007_full__0.98_0.49_0.64",
                # "model_cnn_classify_fold_5_epoch_39_train_loss_0.1243_macro0.7373_full__0.98_0.6_0.63_test_loss_0.1866_macro0.7029_full__0.98_0.5_0.63"
                ],
        },
        "model_6": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_6",
            "list_checkpoint": [
                # "model_cnn_classify_fold_6_epoch_44_train_loss_0.1436_macro0.7262_full__0.98_0.6_0.6_test_loss_0.1955_macro0.7047_full__0.97_0.55_0.59",
                "model_cnn_classify_fold_6_epoch_45_train_loss_0.1402_macro0.7174_full__0.98_0.59_0.58_test_loss_0.1968_macro0.7097_full__0.97_0.56_0.6"
                ],
        },
        "model_7": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_7",
            "list_checkpoint": [
                # "model_cnn_classify_fold_7_epoch_23_train_loss_0.1713_macro0.6666_full__0.98_0.49_0.53_test_loss_0.1982_macro0.6501_full__0.97_0.42_0.55",
                "model_cnn_classify_fold_7_epoch_26_train_loss_0.1595_macro0.6975_full__0.98_0.55_0.57_test_loss_0.1976_macro0.6601_full__0.97_0.46_0.54"
                ],
        },
        "model_8": {
            "type_model": "cnn_classify",
            "folder_model": "/home/trangtv/Documents/project/HateSpeechDectection/module_train/save_model/8_fold_1/fold_8",
            "list_checkpoint": [
                "model_cnn_classify_fold_8_epoch_37_train_loss_0.1561_macro0.7024_full__0.98_0.56_0.56_test_loss_0.1984_macro0.7022_full__0.98_0.48_0.65",
                # "model_cnn_classify_fold_8_epoch_39_train_loss_0.1537_macro0.7144_full__0.98_0.6_0.56_test_loss_0.2004_macro0.6964_full__0.98_0.48_0.64",
                ],
        },
    }

    get_average_predict_model("submit.txt", path_data_test,
                              dict_model)
