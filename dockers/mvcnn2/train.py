import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from Logger import Logger
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from config import get_config, add_to_config

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.system("mkdir -m 777 {}".format(log_dir))
    else:
        print('WARNING: summary folder already exists...')

def train(config):
    print('Starting...')
    pretraining = not config.no_pretraining
    log_dir = config.name
    create_folder(config.name)
  

    print('--------------stage 1--------------')
    # STAGE 1
    log_dir = os.path.join(config.log_dir,config.name+'_stage_1')
    create_folder(log_dir)
    cnet = SVCNN(config, pretraining=pretraining)
    
    optimizer = optim.Adam(cnet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_path = os.path.join(config.data, "*/train")  
    train_dataset = SingleImgDataset(train_path, config, scale_aug=False, rot_aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.stage1_batch_size, shuffle=True, num_workers=0)
    
    val_path = os.path.join(config.data, "*/test")
    val_dataset = SingleImgDataset(val_path, config, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.stage1_batch_size, shuffle=False, num_workers=0)
    
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),config, log_dir, num_views=1)
    trainer.train(config,config.stage1_batch_size)
    #cnet.load(os.path.join(log_dir, config.snapshot_prefix + str(30)))
    
    # STAGE 2
    print('--------------stage 2--------------')
    log_dir = os.path.join(config.log_dir,config.name+'_stage_2')
    create_folder(log_dir)
    cnet_2 = MVCNN(cnet, config)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(train_path,config, scale_aug=False, rot_aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.stage2_batch_size, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(val_path, config, scale_aug=False, rot_aug=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.stage2_batch_size, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), config, log_dir, num_views=config.num_views)
    trainer.train(config,config.stage2_batch_size)
    
    
def test(config):
    log_dir = os.path.join(config.log_dir, config.name+'_stage_2')
  
    val_path = os.path.join(config.data, "*/test")    
    
    val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=config.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.stage2_batch_size, shuffle=False, num_workers=0)

    pretraining = not config.no_pretraining
    cnet = SVCNN(config.name, nclasses=config.num_classes, cnn_name=config.cnn_name, pretraining=pretraining)
    
    cnet_2 = MVCNN(config.name, cnet, nclasses=config.num_classes, cnn_name=config.cnn_name, num_views=config.num_views)
    cnet_2.load(os.path.join(log_dir, config.snapshot_prefix + str(config.weights)))
    optimizer = optim.Adam(cnet_2.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    
    trainer = ModelNetTrainer(cnet_2, None, val_loader, optimizer, nn.CrossEntropyLoss(), config, log_dir, num_views=config.num_views)
    
    labels, predictions = trainer.update_validation_accuracy(config.weights, test=True)
    import Evaluation_tools as et
    eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
    et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
    et.make_matrix(config.data, eval_file, config.log_dir)
    
    
if __name__ == '__main__':
    config = get_config()
    if not config.test:
        train(config)
        if config.weights == -1:
            config = add_to_config(config, 'weights', config.max_epoch)
        else:
            config = add_to_config(config, 'weights', config.max_epoch + config.weights)
        config = add_to_config(config, 'test', True)        
        
    test(config)

        



