# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from PIL import Image
import time
import os
from model import *
from model_AR import AR
from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import pdb
import scipy.io as sio
import losses
import random
version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../../datasets/Market-1501/tmp_2/',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--emb_size', default=128, type=int, help='text_emb_size')
parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')
parser.add_argument('--resume', default=False, type=bool, help='resume the training')
parser.add_argument('--resume_epoch', default=0, type=int, help='epoch which start epoch in resumed training')
parser.add_argument('--DP', default = False, type=bool, help = 'Dataparallel or not')
parser.add_argument('--m', default=0.2, type=float, help='margin')
parser.add_argument('--s', default=12, type=int, help='Proxy_scale')
parser.add_argument('--Pretrained', action='store_true', help='Use attribute classifier pretrained model')
parser.add_argument('--MA_loss', action='store_true', help='Use Modality Alignment loss with ArcFace')
parser.add_argument('--decay', default=10, type=int, help='lr decay epoch')
parser.add_argument('--cuda_seed', default=1, type=int, help='random seed')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--pre_epoch', default='1', type=str, help='Epoch of pretrained model')
parser.add_argument('--lamd', default=4.0, type=float, help='Weight hyper-parameter for ASMR')


opt = parser.parse_args()
num_epoch = opt.num_epoch
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []

seed = opt.seed
cuda_seed = opt.cuda_seed
# torch.manual_seed(seed)
# torch.cuda.manual_seed(cuda_seed)
# # torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True
# torch.backends.cudnn.benchmark = False
# random.seed(seed)
# np.random.seed(seed)

for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

# att_all_train = [['young','teenager','adult','old'],['no', 'backpack', 'bag', 'handbag'],
#                  ['no','downblack','downblue','downbrown','downgray','downgreen','downpink','downpurple','downwhite','downyellow'],
#                  ['no','upblack','upblue','upgreen','upgray','uppurple','upred','upwhite','upyellow'],['dress','pants'],
#                  ['long lower body clothing','short'],['long sleeve','short sleeve'],['short hair','long hair'],['no','hat'],
#                  ['male','female']]


# num_bit = [2, 6, 8, 8, 3, 2, 23, 5, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
num_bit = [4, 4, 10, 9, 2, 2, 2, 2, 2, 2]
# num_bit = [4, 3, 9, 8, 1, 1, 1, 1, 1, 1]
input_att_dim = np.array(num_bit).sum()
######################################################################
# Convert attribute one-hot vector to attribute category vector
# (508,30) to (508,10)
#---------------------------
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    if nb_classes == 2:
        tmp = torch.zeros(len(T),nb_classes)
        for i in range(len(T)):
            tmp[i,int(T[i])] = 1
        T = tmp
    else:
        import sklearn.preprocessing
        T = sklearn.preprocessing.label_binarize(
            T, classes = range(0, nb_classes)
        )
    T = torch.FloatTensor(T)
    return T


def convert_att_category_Market(att_keys):
    converted_categories = []
    for att in att_keys:
        tmp1 = att.split('_')
        att_category = list(map(int,tmp1))
        converted_categories.append(att_category)

    return converted_categories

def convert_one_hot_Market(att_keys):
    converted_one_hots = []
    for att in att_keys:
        tmp1 = att.split('_')
        ar_label = list(map(int,tmp1))
        converted_one_hot = []
        for i in range(len(ar_label)):
            tmp_i = binarize(torch.tensor([ar_label[i]]), num_bit[i])
            converted_one_hot.append(tmp_i)
        
        converted_one_hot_tensor = torch.cat(converted_one_hot,1).squeeze(0)
        
        converted_one_hots.append(converted_one_hot_tensor)
    converted_one_hots = torch.cat(converted_one_hots,0).reshape(-1,int(np.array(num_bit).sum()))
    return converted_one_hots


# transform_train_list = [
#         #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
#         transforms.Resize((128,64), interpolation=3),
#         transforms.Pad(10),
#         transforms.RandomCrop((128,64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]

# transform_val_list = [
#         transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]

transform_train_list = [
    transforms.Resize((256,128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    transforms.Resize((256,128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
# image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
#                                           data_transforms['val'])

# attribute_datasets['no']=attribute_datasets['no'].astype('float32')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}


train_att_keys = list(image_datasets['train'].class_to_idx.keys())
att_one_hots = convert_one_hot_Market(train_att_keys)
att_category_label =  convert_att_category_Market(train_att_keys)
# class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()


att_category_label = torch.tensor(att_category_label)
diff = torch.stack([torch.stack([att_category_label[i]!=att_category_label[j] for j in range(len(att_category_label))]) for i in range(len(att_category_label))])
diff = Variable(diff.cuda().detach())



since = time.time()
inputs, classes = next(iter(dataloaders['train']))

print(time.time()-since)
######################################################################
# Training the model
# ------------------

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(opt.resume_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                if opt.DP and opt.resume==False:
                    modules = model.module.model.modules()
                elif opt.DP and opt.resume:
                    modules = model.module.modules()
                else:
                    modules = model.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
#                 if epoch < 10:
#                     model.text_embedding.train(False)

#                 att_model.train(True)  # Set model to training mode

            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_T = 0.0
            running_loss_V = 0.0
            running_loss_sim = 0.0
            running_corrects = 0.0
            running_rank1 = 0.0
            # Iterate over data.
            pbar = tqdm(dataloaders[phase])
            count = 0
            att_one_hot = convert_one_hot_Market(train_att_keys)
            for data in pbar:
                # get the inputs
                count += 1
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape

                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    att_one_hot = Variable(att_one_hot.cuda().detach())

                else:
                    inputs, labels, att_one_hot = Variable(inputs), Variable(labels), Variable(att_one_hot)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs_V, outputs_T = model(inputs, att_one_hot)
#                         txt_outputs = att_model(att_w2v)

                else:
                    outputs_V, outputs_T = model(inputs, att_one_hot)
                att_V = {}
                att_T = {}
                sm = nn.Softmax(dim=1)

                num_att = 10
                score_T = {}
                score_V = 0
                loss_V = 0
                loss_T = 0
                V_preds = {}
                T_preds = {}

                cos, loss_arc = criterion(outputs_V, outputs_T, labels)
                cos_mu, rw, W = regularizer(outputs_T, diff)


                loss = loss_arc + opt.lamd * rw

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':                    
                    loss.backward()
                    optimizer.step()
                # statistics
                rank1 = float(torch.sum(torch.LongTensor([int(cos[i].argmax()) for i in range(int(cos.shape[0]))]).cuda() == labels))
                running_rank1 += rank1
#                 running_corrects += float(torch.sum(preds == labels))
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: \033[93m{:.6f}\033[0m  Cos_mu: \033[91m{:.6f}\033[0m Acc: \033[92m{:.3f}%\033[0m'.format(
                    epoch + 1, count + 1, len(dataloaders[phase]),
                    100. * count / len(dataloaders[phase]),
                    loss_arc.item(),
                    cos_mu,
                    100.*(rank1/now_batch_size)))




            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_rank1 = running_rank1 / dataset_sizes[phase]
            epoch_acc = 0
            print('{} Loss: {:.4f} R@1: {:.4f}'.format(
                phase, epoch_loss, epoch_rank1))
            print(W.squeeze(1))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_rank1)
            # deep copy the model
            if phase == 'train':
                if epoch >= 0:
                    save_network(model, epoch+1)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
#     model.load_state_dict(last_model_wts)
#     save_network(model, 'last')
    return model


######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])
######################################################################
# load model for resume
#---------------------------
def load_network(network):
    load_path = os.path.join('./model',name,'net_%s.pth'%opt.resume_epoch)
    network.load_state_dict(torch.load(load_path))
    return network
def load_pretrained_network(network):
    load_path = './model/Pretrained_AR_Market/net_'+ opt.pre_epoch+'.pth'
    network.load_state_dict(torch.load(load_path))
    return network
######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#



num_cat =  [4, 4, 10, 9, 2, 2, 2, 2, 2, 2]
if opt.Pretrained:
    model_structure = AR(num_cat)
    model = load_pretrained_network(model_structure)
    if opt.emb_size < 256:
        model = ATT_proxy_pretrained_one_hot_128(model, input_att_dim, opt.emb_size)
    else:
        model = ATT_proxy_pretrained_one_hot_512(model, input_att_dim, opt.emb_size)
else:
    model = ATT_proxy_one_hot_128(input_att_dim, opt.emb_size)

regularizer = losses.ASMR(len(train_att_keys),len(num_bit)).cuda()

opt.nclasses = len(train_att_keys)
# print(model)


ignored_params = []
ignored_params = list(map(id, model.text_embedding.parameters() ))
params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
        {'params': params, 'lr': opt.lr},
        {'params': model.text_embedding.parameters(), 'lr': 15*opt.lr},
        {'params': regularizer.weight, 'lr': 5*opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.decay, gamma=0.1)



######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU (Titan XP).
#
dir_name = os.path.join('./model',name)

if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')
copyfile('./losses.py', dir_name+'/losses.py')
# pdb.set_trace()
opt.cuda_seed = torch.cuda.initial_seed()
opt.seed = torch.initial_seed()
# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()

if opt.resume:
    print("loading checkpoint epoch %s" %opt.resume_epoch)
    if opt.DP:
        model = load_network(nn.DataParallel(model)).module
    model = load_network(model)
    print("%s epoch is loaded" %opt.resume_epoch)
if opt.DP:
#     pdb.set_trace()
    model = nn.DataParallel(model)


criterion_CE = nn.CrossEntropyLoss()
if opt.MA_loss:
    criterion = losses.MA_ArcFace(opt.s, opt.m).cuda()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epoch)

