import picklefrom torchtext import dataimport torchfrom torchtext.vocab import Vectorsclass MyPretrainedVector(Vectors):    def __init__(self, name_file, cache):        super(MyPretrainedVector, self).__init__(name_file, cache=cache)def load_text_label(path_file, is_train=True):    list_text_data = []    list_label = []    with open(path_file, "r") as rf:        for e_line in rf.readlines():            arr_line = e_line.replace("\n", "").split("|")            list_text_data.append(arr_line[0])            if is_train:                list_label.append(arr_line[1])        if is_train:            return list_text_data, list_label        else:            return list_text_datadef load_data_ml(path_file, is_train=True):    with open(path_file, "rb") as rf:        if is_train:            x, y = pickle.load(rf)            return x, y        else:            x = pickle.load(rf)            return xdef load_data_word_lstm_char(path_file_data,                             name_file_train,                             name_file_test=None,                             min_freq_word=1,                             min_freq_char=1,                             batch_size=2,                             cache_folder=None,                             name_vocab=None):    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)    inputs_char = data.NestedField(inputs_char_nesting,                                   init_token="<bos>", eos_token="<eos>")    labels = data.LabelField(sequential=False)    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)), ('labels', labels)])    if name_file_test is not None:        train, test = data.TabularDataset.splits(path=path_file_data,                                                 train=name_file_train,                                                 test=name_file_test,                                                 fields=tuple(fields),                                                 format='csv',                                                 skip_header=False,                                                 csv_reader_params={'delimiter': '|'})        if cache_folder is not None and name_vocab is not None:            inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word,                                    vectors=[MyPretrainedVector(name_vocab, cache_folder)])        else:            inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word)        inputs_char.build_vocab(train.inputs_char, test.inputs_char, min_freq=min_freq_char)        labels.build_vocab(train.labels)        train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),                                                           batch_size=batch_size,                                                           shuffle=True,                                                           device=torch.device("cuda:0"                                                                               if torch.cuda.is_available() else "cpu"))        dict_return = {'iters': (train_iter, test_iter),                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}    else:        path_file_data_train = path_file_data + name_file_train        train = data.TabularDataset(path_file_data_train,                                    fields=tuple(fields),                                    format='csv',                                    skip_header=False,                                    csv_reader_params={'delimiter': '|'})        if cache_folder is not None and name_vocab is not None:            inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word,                                    vectors=[MyPretrainedVector(name_vocab, cache_folder)])        else:            inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word)        inputs_char.build_vocab(train.inputs_char, min_freq=min_freq_char)        labels.build_vocab(train.labels)        train_iter = data.BucketIterator(train,                                         batch_size=batch_size,                                         shuffle=True,                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))        dict_return = {'iters': (train_iter),                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}    return dict_return# load data specify for model use char and word language modeldef load_data_lm_word_char(path_file_data,                           name_file_train,                           name_file_test=None,                           min_freq_word=1,                           min_freq_char=1,                           batch_size=2,                           cache_folder=None,                           name_vocab=None):    tokenize_word = lambda x: x.split()    inputs_word = data.Field(tokenize=tokenize_word, init_token="<bos>", eos_token="<eos>", batch_first=True)    inputs_char = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", batch_first=True)    labels = data.LabelField(sequential=False)    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)), ('labels', labels)])    if name_file_test is not None:        train, test = data.TabularDataset.splits(path=path_file_data,                                                 train=name_file_train,                                                 test=name_file_test,                                                 fields=tuple(fields),                                                 format='csv',                                                 skip_header=False,                                                 csv_reader_params={'delimiter': '|'})        if cache_folder is not None and name_vocab is not None:            inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word,                                    vectors=[MyPretrainedVector(name_vocab, cache_folder)])        else:            inputs_word.build_vocab(train.inputs_word, test.inputs_word, min_freq=min_freq_word)        inputs_char.build_vocab(train.inputs_char, test.inputs_char, min_freq=min_freq_char)        labels.build_vocab(train.labels)        train_iter, test_iter = data.BucketIterator.splits(datasets=(train, test),                                                           batch_size=batch_size,                                                           shuffle=True,                                                           device=torch.device("cuda:0"                                                           if torch.cuda.is_available() else "cpu"))        dict_return = {'iters': (train_iter, test_iter),                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}    else:        path_file_data_train = path_file_data + name_file_train        train = data.TabularDataset(path=path_file_data_train,                                    fields=tuple(fields),                                    format='csv',                                    skip_header=False,                                    csv_reader_params={'delimiter': '|'})        if cache_folder is not None and name_vocab is not None:            inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word,                                    vectors=[MyPretrainedVector(name_vocab, cache_folder)])        else:            inputs_word.build_vocab(train.inputs_word, min_freq=min_freq_word)        inputs_char.build_vocab(train.inputs_char, min_freq=min_freq_char)        labels.build_vocab(train.labels)        train_iter = data.BucketIterator(dataset=train,                                         batch_size=batch_size,                                         shuffle=True,                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))        dict_return = {'iters': (train_iter),                       'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)}    return dict_returnif __name__ == '__main__':    path_file = "../dataset/data_for_train/exp_test.pkl"    load_data_ml(path_file)