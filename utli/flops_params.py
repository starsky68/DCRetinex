import torch
from thop import profile
from model.PINet import PINet

model = PINet()
input = torch.randn(1, 3, 128, 128) # 960，720
flops, params = profile(model, inputs=(input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')