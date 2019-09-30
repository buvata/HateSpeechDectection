import torch
import torch.nn as nn
from torch.nn import functional as F
from module_train.model_architecture_dl.sub_layer.cnn_feature_extract import CNNFeatureExtract1D , CNNFeatureExtract2D
import os
import json


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class CNNClassifyWordCharNgram(nn.Module):
    def __init__(self, cf, vocabs):
        super(CNNClassifyWordCharNgram, self).__init__()
        vocab_word, vocab_char, vocab_label = vocabs

        self.vocabs = vocabs
        self.cf = cf
        self.filter_num_word = cf['filter_num_word']
        self.filter_num_char = cf['filter_num_char']
        self.kernel_sizes_char = cf['kernel_sizes_word']
        self.kernel_sizes_word = cf['kernel_sizes_char']
        self.dropout_cnn = cf['dropout_cnn']
        self.output_size = len(vocab_label)

        len_feature_extract = 0
        len_feature_extract_char = 0
        len_feature_extract_word = 0

        self.word_embedding_dim = cf['word_embedding_dim']
        self.word_embedding_layer = nn.Embedding(len(vocab_word), self.word_embedding_dim)
        if vocab_word.vectors is not None:
            if self.word_embedding_dim != vocab_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(self.word_embedding_dim,
                                                                               vocab_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocab_word.vectors)
            self.word_embedding_layer.requires_grad = False

        if cf['D_cnn'] == "1_D":
            self.cnn_extract_word_ft = CNNFeatureExtract1D(self.word_embedding_dim,
                                                           self.filter_num_word,
                                                           self.kernel_sizes_word,
                                                           self.dropout_cnn)
        else:
            self.cnn_extract_word_ft = CNNFeatureExtract2D(self.word_embedding_dim,
                                                           self.filter_num_word,
                                                           self.kernel_sizes_word,
                                                           self.dropout_cnn)

        len_feature_extract_word += len(self.kernel_sizes_word) * self.filter_num_word

        self.char_embedding_dim = cf['char_embedding_dim']
        if self.char_embedding_dim != 0:
            self.char_embedding_layer = nn.Embedding(len(vocab_char), self.char_embedding_dim)
            if cf['D_cnn'] == "1_D":
                self.cnn_extract_char_ft = CNNFeatureExtract1D(self.char_embedding_dim,
                                                               self.filter_num_char,
                                                               self.kernel_sizes_char,
                                                               self.dropout_cnn)
            else:
                self.cnn_extract_char_ft = CNNFeatureExtract2D(self.char_embedding_dim,
                                                               self.filter_num_char,
                                                               self.kernel_sizes_char,
                                                               self.dropout_cnn)

            len_feature_extract_char += len(self.kernel_sizes_char) * self.filter_num_char

            len_feature_extract = len_feature_extract_char + len_feature_extract_word

        self.dropout_ffw = nn.Dropout(cf['dropout_ffw'])

        self.label = nn.Linear(len_feature_extract, len(vocab_label))

    def compute(self, batch):
        inputs_word_emb = self.word_embedding_layer(batch.inputs_word)
        word_ft = self.cnn_extract_word_ft(inputs_word_emb)

        ft = self.dropout_ffw(word_ft)
        if self.char_embedding_dim != 0:
            inputs_char_emb = self.char_embedding_layer(batch.inputs_char)
            char_ft = self.cnn_extract_char_ft(inputs_char_emb)
            ft = self.dropout_ffw(torch.cat([word_ft, char_ft], -1))
        return ft

    def forward(self, batch):
        with torch.no_grad():
            output_ft = self.compute(batch)
            output_predictions = self.label(output_ft)

        return output_predictions

    def loss(self, batch):
        target = batch.labels

        output_ft = self.compute(batch)
        logits = self.label(output_ft)
        # class_weights = torch.FloatTensor([0.1, 0.9, 0.8]).cuda()
        loss = F.cross_entropy(logits, target)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target

    @classmethod
    def create(cls, path_folder_model, cf, vocabs):
        model = cls(cf, vocabs)
        if cf['use_xavier_weight_init']:
            model.apply(xavier_uniform_init)

        if torch.cuda.is_available():
            model = model.cuda()

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
        print(checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)


if __name__ == '__main__':
    pass
