from tqdm import tqdm
import torch
import torch.optim as optim

import os


class Trainer(object):
    def __init__(self, path_save_model, model, cf_model, prefix_model, log_file, train_iter, test_iter=None):
        self.path_save_model = path_save_model
        self.model = model
        self.cf_model = cf_model
        self.prefix_model = prefix_model

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
                total_correct = 0
                total_sample = 0
                count = 0

                prog_iter = tqdm(self.train_iter, leave=False)
                for batch in prog_iter:
                    self.model.train()
                    self.optimizer.zero_grad()

                    loss, correct, sample = self.model.loss(batch)

                    total_correct += correct
                    total_sample += sample

                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    count += 1

                    prog_iter.set_description('Trainning')
                    prog_iter.set_postfix(loss=(epoch_loss / count))

                total_loss = round(epoch_loss / count, 4)
                accuracy = round(total_correct / total_sample, 4)

                output_train = {"loss": total_loss,
                                "accuracy": accuracy}
                name_model = "{}_epoch_{}_train_acc_{}_loss_{}".format(self.prefix_model,
                                                                       epoch,
                                                                       output_train['accuracy'],
                                                                       output_train['loss'])
                if self.test_iter is not None:
                    output_test = self.evaluator(self.test_iter)
                    name_model = "{}_test_acc_{}_loss_{}".format(name_model,
                                                                 output_test['accuracy'],
                                                                 output_test['loss'])
                log_report = "\n" + name_model
                print(log_report)
                wf.write(log_report)

                self.model.save(self.path_save_model, name_model)

    def evaluator(self, test_iter):
        total_correct = 0
        total_sample = 0
        total_loss = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_iter:

                loss, correct, sample = self.model.loss(batch)
                total_correct += correct
                total_sample += sample
                total_loss += loss.item()
                count += 1

        final_accuracy = round(total_correct / total_sample, 4)
        final_loss = round(total_loss / count, 4)
        return {'loss': final_loss,
                'accuracy': final_accuracy}
