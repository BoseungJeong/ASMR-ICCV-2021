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
import time
import os
from model_AR import *
from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile

import losses

import pdb
import random
import numpy as np
from tqdm import tqdm

version =  torch.__version__
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='Pretrained_AR_Market', type=str, help='output model name')
parser.add_argument('--data_dir',default='../../datasets/Market-1501/tmp_2',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--AR', action='store_true', help='use attribute recognition')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


######################################################################
# Make diff mat between attribute sets on PA100K dataset(JC)
# Todo (JC) use mat multiplication istead dup of for-loop.
#---------------------------
def make_att_diff_mat(att_list):

    att_len_dict = {1:3, 4:3, 13:2, 15:4, 19:2, 22:3}                 
    start_idx = [0,1,4,7,8,9,10,11,12,13,15,19,21,22,25]

    att_diff_mat = []
    for att_f in tqdm(att_list):
        att_f = torch.IntTensor(list(map(int,list(att_f))))
        att_diff_list=[]
        for att_s in att_list:
            att_s = torch.IntTensor(list(map(int,list(att_s))))
            diff = att_f != att_s
            diff_val = 0
            for idx in start_idx:
                if idx in att_len_dict:
                    att_len = att_len_dict[idx]
                    if sum(diff[idx:idx+att_len]) > 0:
                        diff_val += 1
                else:
                    diff_val += diff[idx]
                if diff_val < 0 or diff_val > 15:
                    print("diff_att_mat error!")
                    pdb.set_trace()
            att_diff_list.append(diff_val)
        att_diff_mat.append(att_diff_list)
    
    att_diff_mat = np.stack(att_diff_mat)
    att_diff_mat = torch.FloatTensor(att_diff_mat)

    return att_diff_mat

def convert_ar_label_Market(att_keys):
    converted_labels = []
    for att in att_keys:
        tmp1 = att.split('_')
        ar_label = list(map(int,tmp1))
        converted_labels.append(ar_label)

    return converted_labels


######################################################################
# Load Data
# ---------
#

"""
transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
"""
transform_train_list = [
    #transforms.Resize((384,192), interpolation=3),
    transforms.Resize((256,128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    #transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
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
#image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
#                                         data_transforms['val'])

#balanced_sampler = sampler.BalancedSampler(image_datasets['train'], batch_size=opt.batchsize, images_per_class=)

train_att_keys = list(image_datasets['train'].class_to_idx.keys())
all_ar_labels = convert_ar_label_Market(train_att_keys)
all_ar_labels = torch.LongTensor(all_ar_labels)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train']}
              #for x in ['train', 'val']}
#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs_tmp, classes_tmp = next(iter(dataloaders['train']))
print(time.time()-since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        #for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                # JC: att_keys
                att_keys = train_att_keys
                # JC: batchnorm freeze
                """
                modules = model.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                """
            else:
                model.train(False)  # Set model to evaluate mode
                #att_keys = val_att_keys

            running_loss = 0.0
            running_corrects = [0 for i in range(len(num_cat))]

            #att_onehot_vectors = [list(map(int,list(att_keys[idx]))) for idx in range(len(att_keys))]

            #att_inputs = torch.FloatTensor(att_onehot_vectors)
            #converted_labels, num_cat = convert_onehot_to_ar_PETA(att_inputs)
            #converted_labels = torch.stack(converted_labels,0).t()

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                ###JC convert labels to text_att and make negative samples and matching labels

                """
                att_onehot_neg_vectors = []
                dup_label_list = []

                if phase == 'train':
                    for dup_stage in range(opt.negR-1):
                        for i in range(labels.shape[0]):
                            if dup_stage == 0:
                                dup_label_list.append(labels[i])
                            neg_label = labels[i]
                            while neg_label in dup_label_list:
                                neg_label = random.randint(0, len(list(att_keys))-1)
                            dup_label_list.append(neg_label)
                            att_onehot_neg_vectors.append(list(map(int,list(att_keys[neg_label]))))
                else:
                    for dup_stage in range(opt.negR-1):
                        for i in range(labels.shape[0]):
                            neg_label = labels[i]
                            while neg_label == labels[i]:
                                neg_label = random.randint(0, len(list(att_keys))-1)
                            dup_label_list.append(neg_label)
                            att_onehot_neg_vectors.append(list(map(int,list(att_keys[neg_label]))))

                for idx, att_n in enumerate(att_onehot_neg_vectors):
                    if att_n == att_onehot_vectors[idx%labels.shape[0]]:
                        print("negative error")
                        pdb.set_trace()
                """

                ar_labels = all_ar_labels[labels]
                ar_labels = ar_labels.t()
                ###
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    ar_labels = Variable(ar_labels.cuda().detach())
                else:
                    inputs,  ar_labels = Variable(inputs), Variable(ar_labels)
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)
                num_att = len(ar_labels)
                loss = 0
                preds = []

                for i in range(num_att):
                    loss += criterion(outputs[i], ar_labels[i])
                    score = sm(outputs[i])
                    preds.append(torch.max(score.data,1))
                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size

                for i in range(len(num_cat)):
                    running_corrects[i] += float(torch.sum(preds[i][1] == ar_labels[i].data))

            avg_epoch_acc = 0
            for i in range(len(num_cat)):
                epoch_acc = running_corrects[i] / (dataset_sizes[phase])
                avg_epoch_acc += epoch_acc
                print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            avg_epoch_acc = avg_epoch_acc / float(len(num_cat))
            epoch_loss = running_loss / (dataset_sizes[phase])
            print('{} Loss: {:.4f} Total Acc: {:.4f}'.format(
                phase, epoch_loss, avg_epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-avg_epoch_acc)            
            # deep copy the model
            """
            if phase == 'val':
                last_model_wts = model.state_dict()
                #if epoch%10 == 1 or epoch%10 == 2 or epoch%10 == 3:
                #if epoch%10 == 9:
            """
            if epoch >= 0:
                save_network(model, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Make negative samepls with one-hot labels on PA100K dataset (JC)
#---------------------------
def make_negative_with_onehot(att_labels, num_changes):
    change_list = []
    while not len(change_list)==num_changes:
        rand_val = random.randint(0,14)
        if not rand_val in change_list:
            change_list.append(rand_val)
    change_list.sort()

    att_len_dict = {1:3, 2:3, 9:2, 10:4, 11:2, 13:3}
    start_idx = [0,1,4,7,8,9,10,11,12,13,15,19,21,22,25]
    temp = att_labels.copy()

    for val in change_list:
        start = start_idx[val]
        if not val in att_len_dict:
            temp[start] = int(not temp[start])
        else:
            end = start + att_len_dict[val]
            if sum(temp[start:end]) == 1:
                true_idx = temp[start:end].index(1) + start
                temp[true_idx] = 0
                change_idx = true_idx
                while change_idx == true_idx:
                    change_idx = random.randint(start,end-1)
                temp[change_idx] = 1
            else:
                change_idx = random.randint(start,end-1)
                temp[change_idx] = 1
    return temp

######################################################################
# Convert proxy one-hot vectors to attribute recognition labels from PA100K dataset (JC)
#---------------------------
def convert_onehot_to_ar_PA100K(one_hot_labels):
    start_idx = [0,1,4,7,8,9,10,11,12,13,15,19,21,22,25]
    ar_labels = []

    for idx, s_idx in enumerate(start_idx):
        cat_label = torch.LongTensor()
        for label in one_hot_labels:
            if not s_idx == start_idx[-1]:
                att_len = start_idx[idx+1] - s_idx
                non_zero_idx = torch.nonzero(label[s_idx:s_idx+att_len])
                if att_len > 1 and len(non_zero_idx) == 1:
                    cat_label = torch.cat((cat_label, non_zero_idx[0]+1),0)
                elif att_len > 1 and len(non_zero_idx) == 0:
                    cat_label = torch.cat((cat_label, torch.LongTensor([0])),0)
                else:
                    cat_label = torch.cat((cat_label, label[s_idx].type(torch.LongTensor).view(1)),0)
            else:
                cat_label = torch.cat((cat_label, label[s_idx].type(torch.LongTensor).view(1)),0)
        ar_labels.append(cat_label)

    return ar_labels

def convert_onehot_to_ar_PETA(multi_hot_labels):

    convert_cat_list = [[80 ,0, 1, 2, 3], [4, 5, 17, 20, 22, 81, 85, 86, 88, 96, 101], [7, 9], [6, 8], [10, 18, 19, 30, 79, 89, 91],[13, 23, 24, 28, 83, 99], [11, 32, 33, 34, 100, 103], [12, 25, 27, 31, 90, 92, 102], [14, 21, 29, 104], [84, 94, 96], [16,87], [15, 82, 98], [i for i in range(35,46)], [i for i in range(46,57)], [i for i in range(57,68)], [i for i in range(68,79)], [26, 93, 97]]

    converted_labels = []

    num_cat = []

    for convertor in convert_cat_list:
        dec_cat = bin2dec(multi_hot_labels.t()[convertor].t(), len(convertor))
        unique, converted_inverse = torch.unique(dec_cat,sorted=True,return_inverse=True)

        converted_labels.append(converted_inverse)
        num_cat.append(len(unique))

    return converted_labels, num_cat

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
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

# num_cat = [2, 6, 8, 8, 3, 2, 23, 5, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
num_cat = [4, 4, 10, 9, 2, 2, 2, 2, 2, 2]
if opt.AR:
    model = AR(num_cat)

opt.nclasses = len(class_names)
#opt.nclasses = 2

# print(model)

if opt.AR:
    #ignored_params = list(map(id, model.classifier.parameters()))
    #ignored_params = list(map(id, model.att_block.parameters()))
    ignored_params = []
    params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_ft = optim.SGD([
             {'params': params, 'lr': opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.num_epoch)

