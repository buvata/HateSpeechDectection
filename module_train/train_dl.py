from tqdm import tqdm
import torch
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import f1_score


class Trainer(object):
    def __init__(self, path_save_model, model, cf_model, prefix_model, log_file, size_label, train_iter, test_iter=None):
        self.path_save_model = path_save_model
        self.model = model
        self.cf_model = cf_model
        self.prefix_model = prefix_model

        self.size_label = size_label

        self.train_iter = train_iter
        self.test_iter = test_iter

        self.log_file = os.path.join(self.path_save_model, log_file)

        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(model_params,
                                    lr=cf_model['learning_rate'],
                                    weight_decay=cf_model['weight_decay'])

    def train(self, num_epochs=50):
        with open(self.log_file, "w") as wf:
            for epoch in range(num_epochs):
                # train phase
                self.train_iter.init_epoch()
                epoch_loss = 0
                l_total_predict = []
                l_total_target = []
                count = 0

                prog_iter = tqdm(self.train_iter, leave=False)
                for batch in prog_iter:
                    self.model.train()
                    self.optimizer.zero_grad()

                    loss, l_predict, l_target = self.model.loss(batch)

                    l_total_predict.extend(l_predict)
                    l_total_target.extend(l_target)

                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    count += 1

                    prog_iter.set_description('Trainning')
                    prog_iter.set_postfix(loss=(epoch_loss / count))

                total_loss = round(epoch_loss / count, 4)

                average_score_classes = f1_score(l_total_target, l_total_predict, average=None)
                average_score_macro = round(f1_score(l_total_target, l_total_predict, average='macro'), 4)

                string_macro_each_classes = ""
                for f1_each_class in average_score_classes.tolist():
                    string_macro_each_classes += "_" + str(round(f1_each_class, 2))

                output_train = {"loss": total_loss,
                                "average_macro": average_score_macro,
                                "average_class": string_macro_each_classes
                              }

                name_model = "{}_epoch_{}_train_loss_{}_macro{}_full_{}".format(self.prefix_model,
                                                                                epoch,
                                                                                output_train['loss'],
                                                                                output_train['average_macro'],
                                                                                output_train['average_class'])
                if self.test_iter is not None:
                    output_test = self.evaluator(self.test_iter)
                    string_macro_each_classes = ""
                    for f1_each_class in output_test['average_class'].tolist():
                        string_macro_each_classes += "_" + str(round(f1_each_class, 2))

                    name_model = "{}_test_loss_{}_macro{}_full_{}".format(name_model,
                                                                          output_test['loss'],
                                                                          output_test['average_macro'],
                                                                          string_macro_each_classes)
                log_report = "\n" + name_model
                print(log_report)
                wf.write(log_report)

                self.model.save(self.path_save_model, name_model)

    def train_k_fold(self, num_epochs=50):
        with open(self.log_file, "w") as wf:
            for epoch in range(num_epochs):
                epoch_loss = 0
                count = 0
                l_average_score_macro_train = []
                l_average_score_macro_test = []

                l_average_score_class_train = []
                l_average_score_class_test = []

                l_loss_train = []
                l_loss_test = []

                for id_fold, e_train_iter in enumerate(self.train_iter):
                    e_train_iter.init_epoch()

                    l_total_predict = []
                    l_total_target = []

                    prog_iter = tqdm(e_train_iter, leave=False)
                    for batch in prog_iter:
                        self.model.train()
                        self.optimizer.zero_grad()

                        loss, l_predict, l_target = self.model.loss(batch)

                        l_total_predict.extend(l_predict)
                        l_total_target.extend(l_target)

                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        count += 1

                        prog_iter.set_description('Trainning')
                        prog_iter.set_postfix(loss=(epoch_loss / count))

                    total_loss = round(epoch_loss / count, 4)

                    l_loss_train.append(total_loss)
                    average_score_macro_train = round(f1_score(l_total_target, l_total_predict, average='macro'), 4)
                    average_score_classes_train = f1_score(l_total_target, l_total_predict, average=None)

                    l_average_score_macro_train.append(average_score_macro_train)
                    l_average_score_class_train.append(average_score_classes_train)

                    output_test = self.evaluator(self.test_iter[id_fold])

                    l_loss_test.append(output_test['loss'])
                    l_average_score_macro_test.append(output_test['average_macro'])
                    l_average_score_class_test.append(output_test['average_class'])

                    string_macro_each_classes_train = ""
                    for f1_each_class_train in average_score_classes_train:
                        string_macro_each_classes_train += "_" + str(round(f1_each_class_train, 2))

                    string_macro_each_classes_test = ""
                    for f1_each_class_test in output_test['average_class']:
                        string_macro_each_classes_test += "_" + str(round(f1_each_class_test, 2))

                    log_each_fold = "epoch_{}_fold_{}_train_loss_{}_macro{}_full_{}".format(epoch,
                                                                                            id_fold,
                                                                                            total_loss,
                                                                                            average_score_macro_train,
                                                                                            string_macro_each_classes_train)

                    log_each_fold = "{}_test_loss_{}_macro{}_full_{}".format(log_each_fold,
                                                                          output_test['loss'],
                                                                          output_test['average_macro'],
                                                                          string_macro_each_classes_test)
                    print("\n" + log_each_fold)
                    wf.write(log_each_fold)

                loss_average_train = sum(l_loss_train) / len(l_loss_train)
                loss_average_test = sum(l_loss_test) / len(l_loss_test)

                average_macro_train = sum(l_average_score_macro_train) / len(l_average_score_macro_train)
                average_macro_test = sum(l_average_score_macro_test) / len(l_average_score_macro_test)

                average_score_class_train = np.mean(np.array(l_average_score_class_train), axis=0).tolist()
                average_score_class_test = np.mean(np.array(l_average_score_class_test), axis=0).tolist()

                string_macro_each_classes_train = ""
                for f1_each_class_train in average_score_class_train:
                    string_macro_each_classes_train += "_" + str(round(f1_each_class_train, 2))

                string_macro_each_classes_test = ""
                for f1_each_class_test in average_score_class_test:
                    string_macro_each_classes_test += "_" + str(round(f1_each_class_test, 2))

                name_model = "{}_epoch_{}_train_loss_{}_macro{}_full_{}".format(self.prefix_model,
                                                                                epoch,
                                                                                loss_average_train,
                                                                                average_macro_train,
                                                                                string_macro_each_classes_train)

                name_model = "{}_test_loss_{}_macro{}_full_{}".format(name_model,
                                                                      loss_average_test,
                                                                      average_macro_test,
                                                                      string_macro_each_classes_test)
                log_report = "\n" + name_model

                print(log_report)
                wf.write(log_report)

                print("\n ********************************* \n")
                wf.write("\n ********************************* \n")

    def evaluator(self, test_iter):
        l_total_predict = []
        l_total_target = []
        total_loss = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_iter:
                loss, l_predict, l_target = self.model.loss(batch)
                l_total_predict.extend(l_predict)
                l_total_target.extend(l_target)

                total_loss += loss.item()
                count += 1

        final_loss = round(total_loss / count, 4)

        average_score_classes = f1_score(l_total_target, l_total_predict, average=None)
        average_score_macro = round(f1_score(l_total_target, l_total_predict, average='macro'), 4)

        output_test = {"loss": final_loss,
                       "average_macro": average_score_macro,
                       "average_class": average_score_classes}

        return output_test

