import argparse
import json
import math
import os
import random as rn
import shutil
from collections import defaultdict

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.distributions import normal
from torchvision import transforms

import models
from datagen import *
import trainer

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--filename", type=str, help="Input folder containing train data", default='data/clean_data.csv')
    parser.add_argument("-o", "--out-path", type=str, help="output folder", default='outputs/')

    parser.add_argument("-m", "--model", type=str, help="Pre-trained model path", default=None)
 
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument('--lr_g', type=float, default=1e-03)
    parser.add_argument('--lr_dsc', type=float, default=1e-07)

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
    args.noise_dim = 5
    args.steplr = 200
    args.filename = args.filename

    args.filters = [16, 32, 32, 32]
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
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, 'inter')):
        os.makedirs(os.path.join(args.out_path, 'inter'))
    else:
        shutil.rmtree(os.path.join(args.out_path, 'inter'))
        os.mkdir(os.path.join(args.out_path, 'inter'))

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

def train():
    args = initParams()
    
    trainDset = GetDataset(args.filename)
    valDset = GetDataset(args.filename, val=True)

    train_loader = torch.utils.data.DataLoader(trainDset,
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    val_loader = torch.utils.data.DataLoader(valDset,
                                               batch_size=4, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    device_ids = list(range(torch.cuda.device_count()))

    generator = models.GENERATOR(args).to(args.device)
    generator.apply(init_weights)
    generator = nn.DataParallel(generator, device_ids)


    if args.disc_word_len:
        disc_word_len = models.DISCWORDLEN(args).to(args.device)
        disc_word_len.apply(init_weights)
        disc_word_len = nn.DataParallel(disc_word_len, device_ids)
    else:
        disc_word_len = None

    # if args.disc_emo:
    #     disc_emo = models.DISCEMO(args).to(args.device)
    #     disc_emo.apply(init_weights)
    #     disc_emo = nn.DataParallel(disc_emo, device_ids)
    # else:
    #     disc_emo = None

    # if args.model:
    #     generator.load_state_dict(torch.load(os.path.join(args.model, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
    #     print('Generator loaded...')
    # if args.disc_word_len:
    #     disc_word_len.load_state_dict(torch.load(os.path.join(args.model_disc_frame, 'disc_frame.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
    #     print('Disc frame loaded...')
    
    emoTrainer = trainer.emoTrainer(args, 
                         generator=generator,
                         disc_word_len=disc_word_len,
                         train_loader=train_loader,
                         val_loader=val_loader)
    
    if args.pre_train:
        emoTrainer.pre_train()
    else:
        emoTrainer.train()

if __name__ == "__main__":
    train()
