import pickle

from torchtext import data
import torch


def load_text_label(path_file, is_train=True):
    list_text_data = []
    list_label = []
    with open(path_file, "r") as rf:
        for e_line in rf.readlines():
            arr_line = e_line.replace("\n", "").split("|")
            list_text_data.append(arr_line[0])
            if is_train:
                list_label.append(arr_line[1])
        if is_train:
            return list_text_data, list_label
        else:
            return list_text_data


def load_data_ml(path_file, is_train=True):
    with open(path_file, "rb") as rf:
        if is_train:
            x, y = pickle.load(rf)
            return x, y
        else:
            x = pickle.load(rf)
            return x


def load_data_word_lstm_char(path_file_data,
                             name_file_train,
                             name_file_test=None,
                             min_freq_word=1,
                             min_freq_char=1,
                             batch_size=2):

    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting,
                                   init_token="<bos>", eos_token="<eos>")

    labels = data.LabelField(sequential=False)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)), ('labels', labels)])

    if name_file_test is not None:
        train, test = data.TabularDataset.splits(path=path_file_data,
                                                 train=name_file_train,
                                                 test=name_file_test,
                                                 fields=tuple(fields),
                                                 format='csv',
                                                 skip_header=True,
                                                 csv_reader_params={'delimiter': '|'})

        inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word)
        inputs_char.build_vocab(train.inputs_char, test.inputs_char, min_freq=min_freq_char)
        labels.build_vocab(train.labels)

        train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),
                                                           batch_size=batch_size,
                                                           sort_key=lambda x: len(x.inputs_word),
                                                           device=torch.device("cuda:0"
                                                                               if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': (train_iter, test_iter),
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}
    else:
        path_file_data_train = path_file_data + name_file_train
        train = data.TabularDataset(path_file_data_train,
                                    fields=tuple(fields),
                                    format='csv',
                                    skip_header=True,
                                    csv_reader_params={'delimiter': '|'})

        inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word)
        inputs_char.build_vocab(train.inputs_char, min_freq=min_freq_char)
        labels.build_vocab(train.labels)
        train_iter = data.BucketIterator(train,
                                         batch_size=batch_size,
                                         sort_key=lambda x: len(x.inputs_word),
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        dict_return = {'iters': (train_iter),
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}
        
        
    return dict_return


# load data specify for model use char and word language model
def load_data_lm_word_char(path_file_data,
                           name_file_train,
                           name_file_test=None,
                           min_freq_word=1,
                           min_freq_char=1,
                           batch_size=2):

    tokenize_word = lambda x: x.split()
    inputs_word = data.Field(tokenize=tokenize_word, init_token="<bos>", eos_token="<eos>", batch_first=True)

    inputs_char = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)

    labels = data.LabelField(sequential=False)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)), ('labels', labels)])

    if name_file_test is not None:
        train, test = data.TabularDataset.splits(path=path_file_data,
                                                 train=name_file_train,
                                                 test=name_file_test,
                                                 fields=tuple(fields),
                                                 format='csv',
                                                 skip_header=True,
                                                 csv_reader_params={'delimiter': '|'})
       

        inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word)
        inputs_char.build_vocab(train.inputs_char, test.inputs_char, min_freq=min_freq_char)
        labels.build_vocab(train.labels)

        train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),
                                                           batch_size=batch_size,
                                                           sort_key=lambda x: len(x.inputs_word),
                                                           device=torch.device("cuda:0"
                                                           if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': (train_iter, test_iter),
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}
    else:
        path_file_data_train = path_file_data + name_file_train
        train = data.TabularDataset(path=path_file_data_train,
                                    fields=tuple(fields),
                                    format='csv',
                                    skip_header=True,
                                    csv_reader_params={'delimiter': '|'})
        inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word)
        inputs_char.build_vocab(train.inputs_char, min_freq=min_freq_char)
        labels.build_vocab(train.labels)

        train_iter = data.BucketIterator(dataset=train,
                                         batch_size=batch_size,
                                         sort_key=lambda x: len(x.inputs_word),
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        dict_return = {'iters': (train_iter),
                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}
       
    return dict_return


def load_data(path_file_data_train):
    static_feature = data.Field(sequential=True, use_vocab=False, batch_first=True)
    inputs_word = data.Field(sequential=True, use_vocab=True, batch_first=True)
     
    fields = ([('inputs_word',inputs_word),  ('static_feature', static_feature)])
    train = data.TabularDataset(path=path_file_data_train,
                                    fields=tuple(fields),
                                    format='csv',
                                    csv_reader_params={'delimiter': '|'}) 
    
    for batch in train.__iter__():
        print (batch.inputs_word)
        print(batch.static_feature)


    inputs_word.build_vocab(train)

    train_iter = data.BucketIterator(dataset=train,
                                         batch_size=2,
                                         sort_key=lambda x: len(x.inputs_word),
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    for batch in train_iter.__iter__():
        print (batch.inputs_word)
        print(batch.static_feature)


if __name__ == '__main__':
    # path_file = "/home/taibk/Documents/Code_ML/HateSpeechDectection/module_dataset/dataset/data_for_train/test.csv"
    # load_data(path_file) 

    # path_file_data ="/home/taibk/Documents/Code_ML/HateSpeechDectection/module_dataset/dataset/data_for_train/"
    # name_file_train = "exp_train.csv"
    # load_data_lm_word_char(path_file_data,
                        #    name_file_train,
                        #    name_file_test=None,
                        #    min_freq_word=1,
                        #    min_freq_char=1,
                        #    batch_size=2)
            