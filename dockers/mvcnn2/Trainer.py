import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
import time

from Logger import Logger


class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, config, log_dir, num_views=1):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = config.name
        self.log_dir = log_dir
        self.num_views = num_views
        self.model.cuda()

        self.LOSS_LOGGER = Logger("{}_loss".format(config.name))
        self.ACC_LOGGER = Logger("{}_acc".format(config.name))
        
    def train(self, config, batch_size):

        self.model.train()
        for epoch in range(config.max_epoch+1):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # train one epoch
            out_data = None
            in_data = None
            losses = []
            accs = []
            for i, data in enumerate(self.train_loader):
                if self.num_views > 1:
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                losses.append(loss.item())
                
                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                accs.append(acc)

                loss.backward()
                self.optimizer.step()
                
                if i % max(config.train_log_frq/batch_size,1) == 0:
                    acc = np.mean(accs)
                    loss = np.mean(losses)
                    self.LOSS_LOGGER.log(loss, epoch, "train_loss")
                    self.ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                    losses = []
                    accs = []
                    print(log_str)
            
            # evaluation
            with torch.no_grad():
                loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
            
            self.LOSS_LOGGER.log(loss, epoch, "eval_loss")
            self.ACC_LOGGER.log(val_overall_acc, epoch, "eval_accuracy")
            self.ACC_LOGGER.save(self.log_dir)
            self.LOSS_LOGGER.save(self.log_dir)
            self.ACC_LOGGER.plot(dest=self.log_dir)
            self.LOSS_LOGGER.plot(dest=self.log_dir)        

            # save model
            if epoch % config.save_period == 0 or epoch == config.max_epoch:
                best_acc = val_overall_acc
                self.model.save(os.path.join(self.log_dir, config.snapshot_prefix + str(epoch)))
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

    def update_validation_accuracy(self, epoch, test=False):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for k, data in enumerate(self.val_loader, 0):
            if self.num_views > 1:
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target
            
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1

            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]
            all_target += target.tolist()
            all_pred += pred.tolist()

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = float(all_correct_points) / all_points
        val_overall_acc = acc#acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()
        if test:
            return all_target, all_pred
        return loss, val_overall_acc, val_mean_class_acc

