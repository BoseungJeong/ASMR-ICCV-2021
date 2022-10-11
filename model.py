import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import pdb
import numpy as np
import torch.nn.functional as F
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
######################################################################

    
class Text_embedding_one_hot(nn.Module):
    def __init__(self, input_size, output_size):
        super(Text_embedding_one_hot, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.embedding = nn.Sequential(
            nn.Linear(self.input_size, 4 * self.output_size),
            #3000 -> 1500#
#             nn.BatchNorm1d(int(self.output_size/2)),
            nn.ReLU(),
            nn.Linear(4 * self.output_size, self.output_size),
            #1500 -> 1024#
#             nn.BatchNorm1d(self.output_size),
            nn.ReLU(),
            nn.Linear(self.output_size, self.output_size)     
            #1024 -> 1024#
            )
        self.embedding.apply(weights_init_kaiming)
#         self.fc = nn.Linear(self.output_size,self.output_size).apply(weights_init_kaiming)        
#         self.ReLU = nn.LeakyReLU(0.1)
    def forward(self, x):
        cls_x = self.embedding(x)     
#         T_l_emb = self.ReLU(cls_x)
        return cls_x
    
    



class ATT_proxy_one_hot_128(nn.Module):
    def __init__(self, input_att_size, output_size):
        super(ATT_proxy_one_hot_128,self).__init__()
        
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        self.text_embedding = Text_embedding_one_hot(input_att_size, output_size)

        vis_emb = []
        vis_emb += [nn.Linear(2048, 4*output_size)]
        vis_emb += [nn.ReLU()]
        vis_emb += [nn.Linear(4 * output_size, output_size)]
        vis_emb += [nn.ReLU()]        
        vis_emb += [nn.Linear(output_size, output_size)]
        
        vis_emb = nn.Sequential(*vis_emb)
        vis_emb.apply(weights_init_kaiming)
        
        self.vis_emb = vis_emb
    def forward(self, x, att):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.vis_emb(x)
        att = self.text_embedding(att)
        
        return x, att      

class ATT_proxy_pretrained_one_hot_512(nn.Module):
    def __init__(self,model, input_att_size, output_size):
        super(ATT_proxy_pretrained_one_hot_512,self).__init__()
#         model_ft = models.resnet50(pretrained=True)
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
#         pdb.set_trace()
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        self.text_embedding = Text_embedding_one_hot(input_att_size, output_size)
#         self.fusion1 = Fusion(att_emb_size, output_size, 10)
        vis_emb = []
        vis_emb += [nn.Linear(2048, output_size)]
        vis_emb = nn.Sequential(*vis_emb)
        vis_emb.apply(weights_init_kaiming)
        
        self.vis_emb = vis_emb
    def forward(self, x, att):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
#         import pdb; pdb.set_trace()
#         x = F.normalize(x,p=2, dim=1)

        x = self.vis_emb(x)
#         att = merge_att_w2v(att,10)

        att = self.text_embedding(att)
#         att = self.fusion1(att)
#         pdb.set_trace()
        
        return x, att 
    
class ATT_proxy_pretrained_one_hot_128(nn.Module):
    def __init__(self,model, input_att_size, output_size):
        super(ATT_proxy_pretrained_one_hot_128,self).__init__()
#         model_ft = models.resnet50(pretrained=True)
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
#         pdb.set_trace()
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        self.text_embedding = Text_embedding_one_hot(input_att_size, output_size)
#         self.fusion1 = Fusion(att_emb_size, output_size, 10)
        vis_emb = []
        vis_emb += [nn.Linear(2048, 4*output_size)]
#         vis_emb += [nn.BatchNorm1d(2 * output_size)]
        vis_emb += [nn.ReLU()]
        vis_emb += [nn.Linear(4 * output_size, output_size)]
#         vis_emb += [nn.BatchNorm1d(output_size)]
        vis_emb += [nn.ReLU()]        
        vis_emb += [nn.Linear(output_size, output_size)]
        vis_emb = nn.Sequential(*vis_emb)
        vis_emb.apply(weights_init_kaiming)
        
        self.vis_emb = vis_emb
    def forward(self, x, att):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
#         import pdb; pdb.set_trace()
#         x = F.normalize(x,p=2, dim=1)

        x = self.vis_emb(x)
#         pdb.set_trace()
#         att = merge_att_w2v(att,10)

        att = self.text_embedding(att)
#         att = self.fusion1(att)
#         pdb.set_trace()
        
        return x, att     


 
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
#     net = ft_net(751, stride=1)
    att_all_train = [['young','teenager','adult','old'],['no', 'backpack', 'bag', 'handbag'],
                 ['no','downblack','downblue','downbrown','downgray','downgreen','downpink','downpurple','downwhite','downyellow'],
                 ['no','upblack','upblue','upgreen','upgray','uppurple','upred','upwhite','upyellow'],['dress','pants'],
                 ['long lower body clothing','short'],['long sleeve','short sleeve'],['short hair','long hair'],['no','hat'],
                 ['male','female']]
#     net = AIHM(100, input_size=300, emb_size=512, total_class = att_all_train)
#     net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(32, 3, 256, 128))
    input_2 = Variable(torch.FloatTensor(32, 10,300))  
    output,output_2,_ , T_g_emb= net(input, input_2)
    print('net output size:')
    print(len(output))
    print(output[0].shape)
    print(len(output_2))
    print(output_2[0][0])
    print(T_g_emb.shape)
