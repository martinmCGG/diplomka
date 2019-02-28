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

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-log_dir", type=str, help="log dir", default='logs')
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=4)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-save_epoch", type=int, help="save model period", default=5)
parser.add_argument("-max_epoch", type=int, help="train for this many epochs", default=30)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)

parser.add_argument("-data", type=str, default="/data")

parser.add_argument("--test", action='store_true')
parser.add_argument('--weights', default=-1, type=int)

parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.system("mkdir -m 777 {}".format(log_dir))
        os.system("mkdir -m 777 {}".format(log_dir))
    else:
        print('WARNING: summary folder already exists...')
        #shutil.rmtree(log_dir)
        #os.system("mkdir -m 777 {}".format(log_dir))
        

def train(args):
    print('Starting...')
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    
    print('--------------stage 1--------------')
    # STAGE 1
    log_dir = os.path.join(args.log_dir,args.name+'_stage_1')
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    n_models_train = args.num_models*args.num_views

    train_path = os.path.join(args.data, "*/train")
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    val_path = os.path.join(args.data, "*/test")
    val_dataset = SingleImgDataset(val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1, save_period=args.save_epoch)
    trainer.train(args.max_epoch+1)

    # STAGE 2
    print('--------------stage 2--------------')
    log_dir = os.path.join(args.log_dir,args.name+'_stage_2')
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views, save_period=args.save_epoch)
    trainer.train(args.max_epoch+1)

def test(args):
    log_dir = os.path.join(args.log_dir, args.name+'_stage_2')
    n_models_train = args.num_models*args.num_views
    train_path = os.path.join(args.data, "*/train")    
    val_path = os.path.join(args.data, "*/test")    
    
    val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)

    pretraining = not args.no_pretraining
    cnet = SVCNN(args.name, nclasses=40, cnn_name=args.cnn_name, pretraining=pretraining)
    
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    cnet_2.load(log_dir, modelfile = "model-{}.pth".format(args.weights))
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    trainer = ModelNetTrainer(cnet_2, None, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    
    labels, predictions = trainer.update_validation_accuracy(args.weights, test=True)
    import Evaluation_tools as et

    eval_file = os.path.join(log_dir, 'mvcnn2.txt')
    et.write_eval_file(args.data, eval_file, predictions, labels, 'MVCNN2')
    et.make_matrix(args.data, eval_file, log_dir)
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.test:
        test(args)
    else:
        train(args)



