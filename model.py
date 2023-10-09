import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def SinusoidalEmbeddings(ts):
    time_dimension = 64
    half_dim =  time_dimension // 2
    embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim).float() * -embeddings)
    embeddings = ts.view(-1, 1).float() * embeddings.view(1, -1)
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)

    return embeddings

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=0.1)

        self.time_lin = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, ch_out),
        )

    def forward(self,x,t):
        t = self.time_lin(t)
        t = t.view(-1, t.shape[1], 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = x + t

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttU_Net(nn.Module):
    def __init__(self,img_ch=3):
        super(AttU_Net,self).__init__()
        
        self.time_lin = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
        )
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,img_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x,t):
        t = self.time_lin(t)

        # encoding path
        x1 = self.Conv1(x, t)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2, t)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3, t)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4, t)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5, t)

        # decoding + concat path
        d5 = self.Up5(x5)

        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5, t)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4, t)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3, t)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2, t)

        d1 = self.Conv_1x1(d2)

        return d1

n = 64
model = AttU_Net(img_ch=1)
dummy_input = torch.randn(n, 1, 32, 32)
dummy_time = torch.randn(n, 1)
dummy_embedding = SinusoidalEmbeddings(dummy_time)
dummy_output = model(dummy_input, dummy_embedding.view(n,-1))
# print(dummy_output.size())