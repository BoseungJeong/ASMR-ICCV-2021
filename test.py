# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import *
import pdb
import torch.nn.functional as F
from tqdm import tqdm
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../../datasets/Market-1501/tmp_2/',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--DP', default = False, type=bool, help = 'Dataparallel or not')
parser.add_argument('--emb_size', default=128, type=int, help='text_emb_size')
parser.add_argument('--att_emb_size', default=128, type=int, help='text_emb_size')
parser.add_argument('--lamd', default=1.0, type=float, help='Epoch of pretrained model')


opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.stride = config['stride']
opt.DP = config['DP']


if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
    
    
num_bit = [4, 4, 10, 9, 2, 2, 2, 2, 2, 2]
# num_bit = [4, 3, 9, 8, 1, 1, 1, 1, 1, 1]
# num_bit = [2, 6, 8, 8, 3, 2, 23, 5, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
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


    
    
    
######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
        


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery']}

att_query_list = list(image_datasets['gallery'].class_to_idx.keys())

use_gpu = torch.cuda.is_available()



# attribute_datasets['short hair'] = attribute_datasets['short hiar']
# att_all_train = [['young','teenager','adult','old'],['no', 'backpack', 'bag', 'handbag'],
#                  ['no','downblack','downblue','downbrown','downgray','downgreen','downpink','downpurple','downwhite','downyellow'],
#                  ['no','upblack','upblue','upgreen','upgray','uppurple','upred','upwhite','upyellow'],['dress','pants'],
#                  ['long lower body clothing','short'],['long sleeve','short sleeve'],['short hair','long hair'],['no','hat'],
#                  ['male','female']]

# align_index = [0,1,9,8,2,3,4,5,6,7]
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

def prediction_score(x,y):
    sm = nn.Softmax(dim=-1)
    prob = {}
    for i in range(len(x)):
        prob_V, _ = torch.max(sm(x[i]),1)
        prob_T, _ = torch.max(sm(y[i]),1)
        prob[i] = torch.min(prob_V,prob_T)
    return prob
######################################################################
# Extract feature
# ----------------------
   
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output
   
def extract_feature_vis_att(model, dataloaders, att_query_list):
    features_vis = torch.FloatTensor().cuda()
    features_vis_gpu = torch.FloatTensor().cuda()
    att_vec = torch.FloatTensor().cuda()
#     att_vec_train = torch.FloatTensor().cuda()
    vis_labels = torch.LongTensor().cuda()
    vis_labels_gpu = torch.LongTensor().cuda()
    
#     temp_img = torch.FloatTensor(1,3,384,192).zero_().cuda()
    temp_img = torch.FloatTensor(1,3,256,128).zero_().cuda()
    
    temp_img = Variable(temp_img)
    
    for att in tqdm(att_query_list, desc='**query_feature_extracting'):
#         pdb.set_trace()
        att_one_hot = convert_one_hot_Market([att]).squeeze(0)
#         att_one_hot = torch.from_numpy(one_hot_datasets['one_hot'][att_query_list.index(att)]).to().float().view(1,-1)
        att_inputs = Variable(att_one_hot.cuda())
        
        ff_att = torch.FloatTensor(1,opt.emb_size).zero_().cuda()
        
        outputs_T  = model.text_embedding(att_inputs) 
        
        ff_att += outputs_T
        
        att_vec = torch.cat((att_vec, ff_att.cuda()),0)
        
    att_vec = att_vec.data.cpu()
    count = 0 
    
    for data in tqdm(dataloaders, desc='**gallery_feature_extracting**'):
        img, label = data
#         pdb.set_trace()
        n, c, h, w = img.size()
        temp_att = torch.FloatTensor(n,att_one_hot.size(0)).zero_().cuda()
        temp_att = Variable(temp_att)
        count += n 
        ff_vis = torch.FloatTensor(n,opt.emb_size).zero_().cuda()
        input_img = Variable(img.cuda())
        vis_feat, _  = model(input_img, temp_att)
        ff_vis += vis_feat
  
        features_vis_gpu = torch.cat((features_vis_gpu.cuda(), ff_vis.cuda()),0)
        vis_labels_gpu = torch.cat((vis_labels_gpu.cuda(),label.cuda()),0)
    
    features_vis_gpu = features_vis_gpu.data.cpu()
    vis_labels_gpu = vis_labels_gpu.data.cpu()  
    return features_vis_gpu, att_vec, vis_labels_gpu

def cal_cos_sim(model, vis_feat, att_feat):

    cos_mat = torch.nn.functional.linear(l2_norm(att_feat), l2_norm(vis_feat))
    return cos_mat

       
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels
######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.att_emb_size < 256:
    model_structure = ATT_proxy_one_hot_128(input_att_dim, opt.emb_size)
else:
    model_structure = ATT_proxy_one_hot_512(input_att_dim, opt.emb_size)
        
if opt.DP:
    model = load_network(nn.DataParallel(model_structure)).module
else:
    model = load_network(model_structure)

    
# Change to test mode
model = model.eval()

if use_gpu:
    model = model.cuda()
    

# Extract feature
with torch.no_grad():
    vis_feat, att_feat, vis_label = extract_feature_vis_att(model,dataloaders['gallery'],att_query_list)
  
    scores = cal_cos_sim(model, vis_feat, att_feat)
    
result = {'labels': vis_label.tolist()}        
scipy.io.savemat('./model/'+str(opt.name)+'/'+str(opt.which_epoch)+'.mat' ,result)    





#####################################################################

#######################################################################
# Evaluate

def evaluate_att(query_label,scores,gallery_labels):
    scores = scores.cpu()
    scores = scores.numpy()
    gallery_labels = gallery_labels.squeeze(0).numpy()
    # predict index
    index = np.argsort(scores)  
    index = index[::-1]
    good_index = np.argwhere(query_label==gallery_labels)
    
    ap_tmp, CMC_tmp, cnt = compute_mAP(index, good_index)
    return ap_tmp, CMC_tmp, cnt


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc, 0

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc, len(rows_good)

######################################################################
scores = torch.FloatTensor(scores)
labels = torch.IntTensor(result['labels'])

print(labels.shape)
# 838
query_label = [i for i in range(len(att_query_list))]
CMC = torch.IntTensor(labels.shape[0]).zero_()
ap = 0.0
cnt = 0
seen_count = 0
unseen_count = 0
for i in tqdm(range(len(query_label)), desc='**evaluation**'):
    ap_tmp, CMC_tmp, cnt_tmp = evaluate_att(i,scores[i],labels)
    if CMC_tmp[0]==-1:
        continue
        
    cnt += cnt_tmp
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC

print('R@1:\t%.3f \nR@5:\t%.3f \nR@10:\t%.3f \nmAP:\t%.3f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100,(ap/len(query_label))*100))

result_path = os.path.join('./model',name,'result.txt')
with open(result_path,'a+') as f:
    f.write(str(opt.which_epoch)+'_epoch'+'\b'+ 'Rank@1:'+"%.2f"%((CMC[0]*100).item())+'%'+'\b'+'Rank@5:'+"%.2f"%((CMC[4]*100).item())+'%'+'\b'+'Rank@10:'+"%.2f"%((CMC[9]*100).item())+'%'+'\b'+ 'mAP:'+"%.2f"%((ap/len(query_label))*100) +'%' +'\n')



print(opt.name)
result = './model/%s/result.txt'%opt.name

