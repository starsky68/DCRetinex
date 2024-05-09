import torch
import torch.nn as nn   
        
class Basicblock(nn.Module):
    def __init__(self, num):
        super(Basicblock,self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        return self.m(x)

class Colorbranch(nn.Module):
    def __init__(self, num,):
        super(Colorbranch,self).__init__()
        
        self.downd = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        c = self.downd(x)
        return self.upd(c)+x, c

class Detailbranch(nn.Module):
    def __init__(self, num,rate=0.5):
        super(Detailbranch,self).__init__()
        
        self.downd = nn.Sequential(
            nn.Conv2d(num, int(num*rate), 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(int(num*rate), 1, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(1, int(num*rate), 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(int(num*rate), num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        d = self.downd(x)
        return self.upd(d)+x, d

        
class DAM(nn.Module):
    def __init__(self, num):
        super(DAM,self).__init__()
        self.c = Colorbranch(num)
        self.d = Detailbranch(num)
        
    def forward(self, x):
        u1, c= self.c(x)
        u2, d= self.d(x)

        return u1+u2, d*c, d, c

        
class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        self.r = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.l = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 1, 3, 1, 1),
            nn.Sigmoid() 
            )
    def forward(self, x):
        return self.r(x), self.l(x)
        

class DCRetinex(nn.Module):  
    def __init__(self, num=64):
        super(DCRetinex,self).__init__()
        self.fl = Basicblock(num)
        self.da = DAM(num)
        self.head = Head(num)
        
    def forward(self, x):
        x = self.fl(x)
        u,_,_,_ = self.da(x)
        R,L = self.head(u)
        return L, R, x
        
