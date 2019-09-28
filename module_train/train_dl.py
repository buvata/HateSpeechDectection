from tqdm import tqdm
import torch
import torch.optim as optim

import os
from operator import add


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
                l_total_correct = [0] * self.size_label
                l_total_sample = [0] * self.size_label
                count = 0

                prog_iter = tqdm(self.train_iter, leave=False)
                for batch in prog_iter:
                    self.model.train()
                    self.optimizer.zero_grad()

                    loss, l_correct, l_sample = self.model.loss(batch)

                    l_total_correct = list(map(add, l_total_correct, l_correct))
                    l_total_sample = list(map(add, l_total_sample, l_sample))

                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    count += 1

                    prog_iter.set_description('Trainning')
                    prog_iter.set_postfix(loss=(epoch_loss / count))

                total_loss = round(epoch_loss / count, 4)

                output_train = {"loss": total_loss,
                                "correct": "_".join(str(x) for x in l_total_correct),
                                "total": "_".join(str(x) for x in l_total_sample)}

                name_model = "{}_epoch_{}_train_loss_{}_correct_{}_total{}".format(self.prefix_model,
                                                                                   epoch,
                                                                                   output_train['loss'],
                                                                                   output_train['correct'],
                                                                                   output_train['total'])
                if self.test_iter is not None:
                    output_test = self.evaluator(self.test_iter)
                    name_model = "{}_test_loss_{}correct_{}_total_{}".format(name_model,
                                                                             output_test['loss'],
                                                                             output_test['correct'],
                                                                             output_test['total'])
                log_report = "\n" + name_model
                print(log_report)
                wf.write(log_report)

                self.model.save(self.path_save_model, name_model)

    def evaluator(self, test_iter):
        l_total_correct = [0] * self.size_label
        l_total_sample = [0] * self.size_label
        total_loss = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_iter:
                loss, l_correct, l_sample = self.model.loss(batch)
                l_total_correct = list(map(add, l_total_correct, l_correct))
                l_total_sample = list(map(add, l_total_sample, l_sample))
                total_loss += loss.item()
                count += 1

        final_loss = round(total_loss / count, 4)

        return {"loss": final_loss,
                "correct": "_".join(str(x) for x in l_total_correct),
                "total": "_".join(str(x) for x in l_total_sample)}

