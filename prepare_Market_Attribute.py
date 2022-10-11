import scipy.io as sio
import numpy as np
import cv2
from PIL import Image
import glob
import os
import torch
from shutil import copyfile

att_all_train = [['young','teenager','adult','old'],['no', 'backpack', 'bag', 'handbag'],
                 ['no','downblack','downblue','downbrown','downgray','downgreen','downpink','downpurple','downwhite','downyellow'],
                 ['no','upblack','upblue','upgreen','upgray','uppurple','upred','upwhite','upyellow'],['dress','pants'],
                 ['long lower body clothing','short'],['long sleeve','short sleeve'],['short hair','long hair'],['no','hat'],
                 ['male','female']]
att_all_test = [['young','teenager','adult','old'],['no', 'backpack', 'bag', 'handbag'],['dress','pants'],
                ['long lower body clothing','short'],['long sleeve','short sleeve'],['short hair','long hair'],['no','hat'],
                ['male','female'], ['no','upblack','upwhite','upred','uppurple','upyellow','upgray','upblue','upgreen'],
                ['no','downblack','downwhite','downpink','downpurple','downyellow','downgray','downblue','downgreen','downbrown']]

down_color_test = [0,6,8,5,7,2,3,1,4]
up_color_test = [0,6,7,5,3,2,1,4]
data_path = '../../datasets/Market-1501'

def one2dec(b, bits):
    if isinstance(b, np.ndarray):
        b = torch.LongTensor(b)
    mask = 1+torch.arange(bits).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

mkt_mat = sio.loadmat(data_path+'/market_attribute.mat')
market_att = mkt_mat['market_attribute']
market_att_arr = np.array(market_att)

train_att_list =list(market_att_arr[0][0][1][0][0])
train_att = np.array(train_att_list)
train_re_anno_market = np.swapaxes(train_att,2,0).reshape(-1,28)
np.unique(train_re_anno_market[:,:-1].reshape(751,27).astype('int'),axis=0).shape

test_att_list =list(market_att_arr[0][0][0][0][0])
test_att = np.array(test_att_list)
test_re_anno_market = np.swapaxes(test_att,2,0).reshape(-1,28)
np.unique(test_re_anno_market[:,:-1].reshape(750,27).astype('int'),axis=0).shape

train_label_list = []
for i in range(751):
    i_label = [one2dec(train_re_anno_market[:,:-1][i][:1].astype('int')-1,1).item(),one2dec(train_re_anno_market[:,:-1][i][1:4].astype('int')-1,3).item(),
               one2dec(train_re_anno_market[:,:-1][i][4:13].astype('int')-1,9).item(), one2dec(train_re_anno_market[:,:-1][i][13:21].astype('int')-1,8).item(),
               one2dec(train_re_anno_market[:,:-1][i][21:22].astype('int')-1,1).item(), one2dec(train_re_anno_market[:,:-1][i][22:23].astype('int')-1,1).item(),
               one2dec(train_re_anno_market[:,:-1][i][23:24].astype('int')-1,1).item(), one2dec(train_re_anno_market[:,:-1][i][24:25].astype('int')-1,1).item(),
               one2dec(train_re_anno_market[:,:-1][i][25:26].astype('int')-1,1).item(), one2dec(train_re_anno_market[:,:-1][i][26:27].astype('int')-1,1).item()
              ]
    train_label_list.append(i_label)
    
    
test_label_list = []
for i in range(750):
    i_label = [one2dec(test_re_anno_market[:,:-1][i][:1].astype('int')-1,1).item(),one2dec(test_re_anno_market[:,:-1][i][1:4].astype('int')-1,3).item(),
               one2dec(test_re_anno_market[:,:-1][i][18:27][down_color_test].astype('int')-1,9).item(), one2dec(test_re_anno_market[:,:-1][i][10:18][up_color_test].astype('int')-1,8).item(),
               one2dec(test_re_anno_market[:,:-1][i][4:5].astype('int')-1,1).item(), one2dec(test_re_anno_market[:,:-1][i][5:6].astype('int')-1,1).item(),
               one2dec(test_re_anno_market[:,:-1][i][6:7].astype('int')-1,1).item(), one2dec(test_re_anno_market[:,:-1][i][7:8].astype('int')-1,1).item(),
               one2dec(test_re_anno_market[:,:-1][i][8:9].astype('int')-1,1).item(), one2dec(test_re_anno_market[:,:-1][i][9:10].astype('int')-1,1).item()
              ]
    test_label_list.append(i_label)

    
    
    
train_file_name_label = ['_'.join(map(str,train_label_list[i])) for i in range(len(train_label_list))]
test_file_name_label = ['_'.join(map(str,test_label_list[i])) for i in range(len(test_label_list))]


download_path = data_path + '/tmp_2'

train_path = data_path + '/pytorch/train_all'
test_path = data_path + '/pytorch/gallery'
query_path = data_path + '/pytorch/query/'
train_save_path = download_path + '/train_all'
query_save_path = download_path + '/query'
test_save_path = download_path + '/gallery'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)
    os.mkdir(query_save_path)
    
train_cls_to_att = {}
for root, dirs, files in os.walk(train_path):
    dirs.sort()
    files.sort()
    for i, classes in enumerate(dirs):
        if str(classes) == '0000':
            continue
        train_cls_to_att[str(classes)]= train_file_name_label[i]

for root, dirs, files in os.walk(train_path):
    dirs.sort()
    files.sort()
    for name in files:
        if name[:4] == '0000':
            continue
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + root[-4:]+'/' + name
        dst_path = train_save_path + '/' + train_cls_to_att[root[-4:]]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
        
test_cls_to_att = {}
for root, dirs, files in os.walk(test_path):
    dirs.sort()
    files.sort()
    for i, classes in enumerate(dirs):
        if str(classes) == '0000' or str(classes)=='-1':
            continue
        test_cls_to_att[str(classes)]= test_file_name_label[i-2]

for root, dirs, files in os.walk(test_path):
    dirs.sort()
    files.sort()
    for name in files:
        if name[:4] == '0000' or name[:2] == '-1':
            continue
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = test_path + '/' + root[-4:]+'/' + name
        dst_path = test_save_path + '/' + test_cls_to_att[root[-4:]]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
        
query_cls_to_att = {}
for root, dirs, files in os.walk(query_path):
    dirs.sort()
    files.sort()
    for i, classes in enumerate(dirs):
        if str(classes) == '0000' or str(classes)=='-1':
            continue
        query_cls_to_att[str(classes)]= test_file_name_label[i]

for root, dirs, files in os.walk(query_path):
    dirs.sort()
    files.sort()
    for name in files:
        if name[:4] == '0000' or name[:2] == '-1':
            continue
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + root[-4:]+'/' + name
        dst_path = test_save_path + '/' + query_cls_to_att[root[-4:]]
        dst_query_path = query_save_path + '/' + query_cls_to_att[root[-4:]]        
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        if not os.path.isdir(dst_query_path):
            os.mkdir(dst_query_path)            
        copyfile(src_path, dst_path + '/' + name)        
        
