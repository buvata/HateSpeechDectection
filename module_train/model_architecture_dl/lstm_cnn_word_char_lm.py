import torch
import torch.nn as nn
import torch.nn.functional as F
from module_train.model_architecture_dl.sub_layer.cnn_feature_extract import CNNFeatureExtract1D, CNNFeatureExtract2D

import os
import json


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class LSTMCNNWordCharLM(nn.Module):
    def __init__(self, cf, vocabs):
        super(LSTMCNNWordCharLM, self).__init__()
        print(vocabs)
        vocab_word, vocab_char, vocab_label = vocabs
        self.cf = cf
        self.vocabs = vocabs
        self.output_size = len(vocab_label)

        self.word_embedding_dim = cf['word_embedding_dim']
        self.hidden_size_word = cf['hidden_size_word']
        self.dropout_rate = cf['dropout_rate']
        self.use_last_as_ft = cf['use_last_as_ft']

        self.word_embedding_layer = nn.Embedding(len(vocab_word), self.word_embedding_dim)
        if vocab_word.vectors is not None:
            if self.word_embedding_dim != vocab_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(self.word_embedding_dim,
                                                                               vocab_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocab_word.vectors)
            self.word_embedding_layer.requires_grad = False

        # lstm get input shape (seq_len, batch_size, input_dim)
        self.layer_lstm_word = nn.LSTM(self.word_embedding_dim,
                                       self.hidden_size_word,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=True)
        self.output_final = self.hidden_size_word * 2
        if self.use_last_as_ft:
            self.output_final = self.hidden_size_word * 4

        self.char_embedding_dim = cf['char_embedding_dim']
        if self.char_embedding_dim != 0 and vocab_char is not None:
            self.char_cnn_filter_num = cf['char_cnn_filter_num']
            self.char_window_size = cf['char_window_size']
            self.dropout_cnn = cf['dropout_cnn']
            self.char_embedding_layer = nn.Embedding(len(vocab_char), self.char_embedding_dim)

            if cf['D_cnn'] == '1_D':
                self.layer_char_cnn = CNNFeatureExtract1D(self.char_embedding_dim,
                                                    self.char_cnn_filter_num,
                                                    self.char_window_size,
                                                    self.dropout_cnn)
            else:
                self.layer_char_cnn = CNNFeatureExtract2D(self.char_embedding_dim,
                                                          self.char_cnn_filter_num,
                                                          self.char_window_size,
                                                          self.dropout_cnn)
            self.hidden_size_char = self.char_cnn_filter_num * len(self.char_window_size)
            self.output_final += self.hidden_size_char

        self.dropout = nn.Dropout(self.dropout_rate)
        self.hidden_size_reduce = cf['hidden_size_reduce']
        if self.hidden_size_reduce > 0:
            self.ffw_layer = nn.Linear(self.output_final, self.hidden_size_reduce)
            self.label = nn.Linear(self.hidden_size_reduce, self.output_size)
        else:
            self.label = nn.Linear(self.hidden_size_word + self.hidden_size_char, self.output_size)

    @staticmethod
    def attention_net(network_output, final_state):
        attn_weights = torch.bmm(network_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(network_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def compute_forward(self, batch):
        inputs_word_emb = self.word_embedding_layer(batch.inputs_word)
        inputs_word_emb = self.dropout(inputs_word_emb)

        output_hidden_word, (_, _) = self.layer_lstm_word(inputs_word_emb)
        final_hidden_state_word = output_hidden_word[:, -1, :]

        attn_output = self.attention_net(output_hidden_word, final_hidden_state_word)

        final_output = attn_output
        if self.use_last_as_ft:
            final_output = torch.cat((final_output, final_hidden_state_word), dim=-1)
        final_output = self.dropout(final_output)

        if self.char_embedding_dim != 0:
            inputs_char_emb = self.char_embedding_layer(batch.inputs_char)
            inputs_char_emb = self.dropout(inputs_char_emb)

            final_char_output = self.layer_char_cnn(inputs_char_emb)
            final_output = torch.cat([final_output, final_char_output], -1)

        final_output = self.dropout(final_output)
        if self.hidden_size_reduce > 0:
            final_output = self.ffw_layer(final_output)

        final_output = self.dropout(final_output)

        return final_output

    def forward(self, batch):
        with torch.no_grad():
            output_ft = self.compute_forward(batch)
            output_predictions = self.label(output_ft)
        return output_predictions

    def loss(self, batch):
        target = batch.labels

        output_ft = self.compute_forward(batch)
        output_predictions = self.label(output_ft)

        loss = F.cross_entropy(output_predictions, target)

        predict_value = torch.max(output_predictions, 1)[1]

        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target

    @classmethod
    def create(cls, path_folder_model, cf, vocabs, device_set="cuda:0"):
        model = cls(cf, vocabs)
        if cf['use_xavier_weight_init']:
            model.apply(xavier_uniform_init)

        if torch.cuda.is_available():
            device = torch.device(device_set)
            model.to(device)

        path_vocab_file = os.path.join(path_folder_model, "vocabs.pt")
        torch.save(vocabs, path_vocab_file)

        path_config_file = os.path.join(path_folder_model, "model_cf.json")
        with open(path_config_file, "w") as w_config:
            json.dump(cf, w_config)

        return model

    @classmethod
    def load(cls, path_folder_model, path_model_checkpoint):
        path_vocab_file = os.path.join(path_folder_model, 'vocabs.pt')
        path_config_file = os.path.join(path_folder_model, 'model_cf.json')

        if not os.path.exists(path_vocab_file) or \
                not os.path.exists(path_config_file) or \
                not os.path.exists(path_model_checkpoint):
            raise OSError(" 1 of 3 file does not exist")

        vocabs = torch.load(path_vocab_file)
        print(vocabs)
        with open(path_config_file, "r") as r_config:
            cf = json.load(r_config)

        model = cls(cf, vocabs)

        if torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict(torch.load(path_model_checkpoint))
        else:
            model.load_state_dict(torch.load(path_model_checkpoint, map_location=lambda storage, loc: storage))
        return model

    def save(self, path_save_model, name_model):
        checkpoint_path = os.path.join(path_save_model, name_model)
        torch.save(self.state_dict(), checkpoint_path)
