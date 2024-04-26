import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from .utils import DisabledSummaryWriter, log_exceptions
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

configs['test']['save_path'] = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+ str(' ') +configs['experiment']['experiment_name']

if 'tensorboard' in configs['experiment'] and configs['experiment']['tensorboard']:
    timestr = configs['test']['save_path']
    writer = SummaryWriter(log_dir=f'runs/{timestr}')
    configs['test']['tensorboard'] = writer
else:
    writer = DisabledSummaryWriter()

def init_seed():
    if 'reproducible' in configs['experiment']:
        if configs['experiment']['reproducible']:
            seed = configs['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        initial_lr = optim_config['lr']
        # final_lr = optim_config['final_lr']
        # total_epochs = configs['train']['epoch']
        # gamma = (final_lr / initial_lr) ** (1 / total_epochs)
        gamma = optim_config['gamma']
        # warmup_steps = int(configs['train']['epoch'] * 0.4)
        # d_model = configs['model']['item_embedding_size']

        # def lr_lambda(step):
        #     return (d_model ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5))


        if optim_config['name'] == 'adam':
            # self.optimizer = optim.Adam(model.parameters(
            # ), lr=initial_lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=optim_config['weight_decay'])
            self.optimizer = optim.Adam(model.parameters(
            ), lr=initial_lr, weight_decay=optim_config['weight_decay'])
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=gamma, min_lr=1e-6)
            # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90, 120, 150, 180], gamma=0.1)
            # self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
            # self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, model, epoch_idx):
        # Calls the sample_negs method on the training dataset, which might be related to negative sampling in recommendation systems.
        train_dataloader = self.data_handler.train_dataloader
        #todo val loss and train loss are different in model test run where you have both dataset the same. check this.
        # train_dataloader.dataset.sample_negs()
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.train()

        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            if not configs['train']['gradient_accumulation']: 
                self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']) if not isinstance(x, list) 
                                  else torch.stack([t.float().to(configs['device']) for t in x], dim=1), tem))

            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            # self.optimizer.step()
            if configs['train']['gradient_accumulation'] and (i + 1) % configs['train']['accumulation_steps'] == 0:
                # Perform gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            elif not configs['train']['gradient_accumulation']:
                # Perform gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            for loss_name in loss_dict:
                _loss_train = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_train
                else:
                    loss_log_dict[loss_name] += _loss_train
            
            if configs['train']['scheduler']:
                self.scheduler.step(loss_log_dict['rec_loss'])
        
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        string_to_append = "_train"
        train_loss_log_dict = {key + string_to_append: value for key, value in loss_log_dict.items()}

        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, train_loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, train_loss_log_dict, save_to_log=False)

    def evaluate_val_loss(self, model, epoch_idx):

        test_loader = self.data_handler.test_dataloader
        loss_log_dict = {}
        ep_loss = 0
        steps = len(test_loader.dataset) // configs['test']['batch_size']

        model.eval()
        with torch.no_grad():
            for i, tem in enumerate(test_loader):
                batch_data = list(map(lambda x: x.long().to(configs['device']) if not isinstance(x, list) 
                                  else torch.stack([t.float().to(configs['device']) for t in x], dim=1), tem))

                loss, loss_dict = model.cal_loss(batch_data)
                ep_loss += loss.item()
                    
                for loss_name in loss_dict:
                    _loss_train = float(loss_dict[loss_name]) / len(test_loader)
                    if loss_name not in loss_log_dict:
                        loss_log_dict[loss_name] = _loss_train
                    else:
                        loss_log_dict[loss_name] += _loss_train

            writer.add_scalar('Loss/test', ep_loss / steps, epoch_idx)

            string_to_append = "_test"
            test_loss_log_dict = {key + string_to_append: value for key, value in loss_log_dict.items()}

            if configs['test']['log_loss']:
                self.logger.log_loss(epoch_idx, test_loss_log_dict)
            else:
                self.logger.log_loss(epoch_idx, test_loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model):
        total_parameters = model.count_parameters()
        print(f"Total number of parameters in the model: {total_parameters}")
        self.create_optimizer(model)
        train_config = configs['train']

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                self.train_epoch(model, epoch_idx)
                self.evaluate_val_loss(model, epoch_idx)
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch_idx + 1}, Learning Rate: {current_lr}")
                if train_config['train_checkpoints'] and epoch_idx % train_config['save_step'] == 0:
                    self.save_model(model)
            self.test(model)
            self.save_model(model)
            return model
        
        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                self.train_epoch(model, epoch_idx)
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    if now_patience == configs['train']['patience']:
                        break

            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        if configs['test']['train_eval']:
            eval_result, cm_im = self.metric.eval(model, self.data_handler.train_dataloader)
            writer.add_image("confusion_matrix/train", cm_im, epoch_idx)
            for i, k in enumerate(configs['test']['k']):
                for metric in configs['test']['metrics']:
                    writer.add_scalar(f'{metric}_top_{k}/train', eval_result[metric][i], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='train set', epoch_idx=epoch_idx)
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result, cm_im = self.metric.eval(model, self.data_handler.valid_dataloader)
            writer.add_image("confusion_matrix/valid", cm_im, epoch_idx)
            for i, k in enumerate(configs['test']['k']):
                for metric in configs['test']['metrics']:
                    writer.add_scalar(f'{metric}_top_{k}/valid', eval_result[metric][i], epoch_idx)
            # writer.add_scalar('HR/valid', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result, cm_im = self.metric.eval(model, self.data_handler.test_dataloader)
            writer.add_image("confusion_matrix/test", cm_im, epoch_idx)
            for i, k in enumerate(configs['test']['k']):
                for metric in configs['test']['metrics']:
                    writer.add_scalar(f'{metric}_top_{k}/test', eval_result[metric][i], epoch_idx)
            # writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplemented
        
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        
        eval_result, _ = self.metric.eval(model, self.data_handler.train_dataloader, test=True)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Train set')
        configs['test']['data']="test"
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result, _ = self.metric.eval(model, self.data_handler.test_dataloader, test=True)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['experiment']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            timestr = configs['test']['save_path']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestr))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestr)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['experiment']:
            pretrain_path = configs['experiment']['pretrain_path']
            module_path = "/".join(['checkpoint', configs['model']['name'], pretrain_path])
            model.load_state_dict(torch.load(module_path))
            self.logger.log(
                "Load model parameters from {}".format(module_path))
            return model
        else:
            raise KeyError("No module_path in configs['train']")
