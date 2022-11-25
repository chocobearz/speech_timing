import argparse
import json
import os
import random as rn
import shutil

import numpy as np 
import torch
import torch.nn as nn

import models
from datagen import *
import GAN_trainer, IMLE_trainer

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--filename", type=str, help="Input folder containing train data", default='../data/clean_data.csv')
    parser.add_argument("-o", "--out-path", type=str, help="output folder", default='outputs/')
    parser.add_argument("-m", "--model", type=str, help="Pre-trained model path", default='checkpoints/')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument('--lr_g', type=float, default=0.0005)
    parser.add_argument('--lr_dsc', type=float, default=0.0005)
    parser.add_argument("--gpu-no", type=str, help="select gpu", default='0')
    parser.add_argument('--seed', type=int, default=9)

    parser.add_argument('--disc_word_len', type=float, default=1)
    parser.add_argument('--disc_emo', type=float, default=None)
    parser.add_argument('--pre_train', type=bool, default=False)
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    args.batch_size = args.batch_size * max(int(torch.cuda.device_count()), 1)
    args.text_dim = 20
    args.emo_dim = 3
    args.noise_dim = args.text_dim
    args.people_dim = 6

    args.steplr = 200
    args.filename = args.filename
    args.MAX_LEN = 0
    args.criterion = 'Wass'

    args.filters = [8, 16, 16, 16]
    #-----------------------------------------#
    #           Reproducible results          #
    #-----------------------------------------#
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)
    torch.manual_seed(args.seed)
    #-----------------------------------------#
   
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, 'inter')):
        os.makedirs(os.path.join(args.out_path, 'inter'))

    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = torch.cuda.is_available() 
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu") 
    args.kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    return args

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def enableGrad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def imle_train():
    args = initParams()
    
    trainDset = GetDataset(args.filename)
    valDset = GetDataset(args.filename, val=True)

    train_loader = torch.utils.data.DataLoader(trainDset,
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    val_loader = torch.utils.data.DataLoader(valDset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=True,
                                               **args.kwargs)
    device_ids = list(range(torch.cuda.device_count()))

    generator = models.GENERATOR(args).to(args.device)
    generator.apply(init_weights)
    generator = nn.DataParallel(generator, device_ids)

    autoencoder = models.AUTOENCODER(args).to(args.device)
    autoencoder.apply(init_weights)
    autoencoder = nn.DataParallel(autoencoder, device_ids)

    if not args.pre_train:
        autoencoder.load_state_dict(torch.load(os.path.join(args.model, 'autoencoder.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
        print('Autoencoder loaded...')
        imle = models.IMLE(args, autoencoder).to(args.device)
        imle = nn.DataParallel(imle, device_ids)   
        # checkpoint = torch.load(os.path.join(args.model, 'imle.pt'), map_location="cuda" if args.cuda else "cpu")
        # sd = autoencoder.state_dict()
        # for k in imle.state_dict().keys():
        #     if k in sd and sd[k].size() == checkpoint[k].size():
        #         sd[k] = checkpoint[k]
        #     else:
        #         print("Missed: ", k)
        # autoencoder.load_state_dict(sd)
    else:
        imle = None

    emoTrainer = IMLE_trainer.emoTrainer(args, 
                         autoencoder=autoencoder,
                         imle=imle,
                         train_loader=train_loader,
                         val_loader=val_loader)
    
    emoTrainer.collect_variance()

    if args.pre_train:
        emoTrainer.pre_train()
    else:
        emoTrainer.collect_variance()
        emoTrainer.train()
    
    emoTrainer.test()


if __name__ == "__main__":
    #train()
    
    imle_train()
