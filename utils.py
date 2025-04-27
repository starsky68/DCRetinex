import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from teacher_model.URetinexNet.test import Inference as urtxinf
from teacher_model.ZeroDCE.lowlight_test import inference as dceinf

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()  
    return loss

def PUEI_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2)  # PUEI
    return loss

def retinex_Illu_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1)
    retinex_loss = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach()) # 公式14
    Illu_loss = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)  # torch.nn.MSELoss()(L1, max_rgb1)是公式11的第一项，tv_loss(L1)是#式11的第二项
    return retinex_loss + Illu_loss
    
def CD_loss(im1, X1, scale=500):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss * scale
    
def A_loss(L1, X1, R1, scale=0.1):#triplet focal loss
    return scale * torch.nn.MSELoss()(R1, X1) * L1.mean()  # 公式10

def teacherloss(im,R,t_model_type='zdce'):
    if t_model_type=='urtx':
        urtx_inf = urtxinf().cuda()
        enhance,_,R_ = urtx_inf(im)
    elif t_model_type=='zdce':
        enhance = dceinf(im)
    loss = torch.nn.MSELoss()(R, torch.sigmoid(enhance))
    return loss


def SUEILoss(R1, L1, im1, X1):
    loss1 = retinex_Illu_loss(L1, R1, im1, X1)
    loss2 = CD_loss(im1, X1, scale=500)
    loss3 = A_loss(L1, X1, R1)
    return loss1 + loss2 + loss3
    
def PUEILoss(R1, R2, L1, im1, X1):
    loss1 = PUEI_loss(R1, R2)
    loss2 = retinex_Illu_loss(L1, R1, im1, X1)
    loss3 = CD_loss(im1, X1, scale=500)
    loss4 = A_loss(L1, X1, R1)
    return loss1 + loss2 + loss3 + loss4 

def TMGLoss(R1, L1, im1, X1,t_model_type='zdce'):
    loss1 = teacherloss(im1, R1,t_model_type)
    loss2 = retinex_Illu_loss(L1, R1, im1, X1)
    loss3 = CD_loss(im1, X1, scale=500)
    loss4 = A_loss(L1, X1, R1)
    return loss1 + loss2 + loss3 + loss4 

    
    
def joint_RGB_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('RGB',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))      
    return result

def joint_L_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('L',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))   
    return result
