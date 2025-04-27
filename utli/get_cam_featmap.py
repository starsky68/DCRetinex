# coding: utf-8
import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from model import *

def img_preprocess(img_in):
    return ToTensor()(img_in)
    
def save_cam(features,out_path,src_size=(600,400)):
    # 1.2 每个通道对应元素求和
    heatmap = torch.sum(features, dim=1)  # 尺度大小， 如torch.Size([1,45,45])
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap-min_value)/(max_value-min_value)*255
    heatmap = heatmap.detach().numpy().astype(np.uint8).transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
    # 原图尺寸大小
    heatmap = cv2.resize(heatmap, src_size,interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    cv2.imwrite(out_path, heatmap)
    
def gen_cam(model, img,out_path):
    L,R,x,i,p,u,up,ui = model(img)
    #print(features)
    #trg = model.pi.p.downd
    save_cam(ui,out_path)
    #ft = R[0].mean(dim=1).squeeze()
    

if __name__ == '__main__':

    path_img = r'/SICE1_weights/visual/1.JPG'
    ckpt_path = r'/SICE1_weights/PINet/014/best_psnr_PILIELoss.pth'
    output_dir = r'/SICE1_weights/visual/11.JPG'

    # 图片读取；网络加载
    img = Image.open(path_img).convert('RGB')#cv2.imread(path_img, 1)  # H*W*C
    #print(img)
    img_input = img_preprocess(img).unsqueeze(0)
    # model = DCRetinex()
    model = visual()
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    model.eval()
    gen_cam(model,img_input,output_dir)
