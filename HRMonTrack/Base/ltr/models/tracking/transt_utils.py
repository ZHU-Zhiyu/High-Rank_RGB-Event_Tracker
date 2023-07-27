import math
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.autograd import Variable, Function
from knn_cuda import KNN

def gether_neibour(X, indx, sample_n, neigh):
    '''
    Output shape B, P1, n1, C
    '''
    B, C, P = X.shape
    B1, n1, p1 = indx.shape
    assert(B == B1)
    assert(n1 == neigh)
    indx = indx.reshape([B1,1, n1*p1]).repeat([1,C,1])
    feature = torch.gather(X, 2, indx)
    feature = feature.reshape([B, C, n1, p1]).permute([0,3,2,1])
    return feature
class GradNorm(Function):
    @staticmethod
    def forward(ctx, input_x):
        a = torch.clone(input_x.detach())
        b = torch.clone(input_x.detach())

        return a, b
    @staticmethod
    def backward(ctx, grad_a, grad_b):
        grad_a_nrom = (grad_a**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        grad_b_nrom = (grad_b**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        grad_a = grad_a / (grad_a_nrom + 1e-6)
        grad_b = grad_b / (grad_b_nrom + 1e-6)
        grad_min = torch.cat([grad_a_nrom, grad_b_nrom],dim=1)
        grad_min = grad_min.min(dim=1,keepdim = True)[0]
        grad_a = grad_a* grad_min
        grad_b = grad_b* grad_min

        return grad_a+grad_b

def knn(x,x2, k, dilation = 1):
    with torch.no_grad():
        inner = -2*torch.matmul(x.transpose(2, 1), x2)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        xx2 = torch.sum(x2**2, dim=1, keepdim=True)
        pairwise_distance = -xx.transpose(2, 1) - inner - xx2
    
        idx = pairwise_distance.topk(k=k*dilation, dim=-2)[1]   # (batch_size, num_points, k)
        return idx[:,:,::dilation]

class Dynamic_sampling_M(Module):
    def __init__(self,k, dilation = 1):
        super(Dynamic_sampling_M, self).__init__()
        self.k = k
        self.knn_k = KNN(k=k, transpose_mode=False)
        self.knn_1 = KNN(k=1, transpose_mode=False)
    def crop_points(self, sampled, target, length):
        
        dist = sampled[:,:2,:] - target[:,:2,:]
        dist = dist**2
        dist = dist.sum(dim = 1).sqrt()

        # flag in shape B,P
        flag = dist> math.sqrt(2)/length

        sampled = sampled.permute([0,2,1])

        sampled[flag,:] = target.permute([0,2,1])[flag,:]

        sampled = sampled.permute([0,2,1])

        return sampled
    def forward(self, x, s_num, Is_temp=False, Temp_ratio=0):

        B,C,P = x.shape
        if s_num <=P:

            get_neighbour = s_num
            length = math.ceil(math.sqrt(get_neighbour))
            mesh = torch.arange(0,length).float()/length
            mesh = mesh.to(x.device)
            mesh01 = mesh[None,:].repeat([length, 1])
            mesh02 = mesh[:,None].repeat([1, length])
            mesh01 = mesh01.reshape([1,1,length*length])
            mesh02 = mesh02.reshape([1,1,length*length])
            mesh = torch.cat([mesh01,mesh02],dim=1)
            mesh = mesh.repeat([B,1,1])
            PointSample = mesh

            # PointSample = torch.rand(B, 2, get_neighbour).to(x.device)
            Point_x = x[:,:2,:]
            # min_x = Point_x.min(dim=2,keepdim = True)[0]
            # max_x = Point_x.max(dim=2,keepdim = True)[0]
            max_t = x[:,2,:].max(dim=1,keepdim = True)[0]
            # print('shape of max_t:{}'.format(max_t.shape))
            # shape in B, C, P
            PointSample_out = torch.cat([PointSample, max_t[:,None].repeat([1,1,s_num]), torch.zeros([B,1,s_num], device = x.device)], dim = 1)
            # PointSample = PointSample*(max_x - min_x) + min_x
            _, indx = self.knn_1(Point_x.contiguous(),PointSample.contiguous())
            # select_point shape B, P1, n1, C
            select_point = gether_neibour(x, indx, get_neighbour, 1)
            select_point = select_point[:,:,0,:].permute([0,2,1])

            dist = PointSample - select_point[:,:2,:]
            dist = dist**2
            dist = dist.sum(dim = 1).sqrt()

            # flag in shape B,P
            flag = dist> math.sqrt(2)/length

            select_point = select_point.permute([0,2,1])

            select_point[flag,:] = PointSample_out.permute([0,2,1])[flag,:]

            select_point = select_point.permute([0,2,1])

        else:
            rand_num = torch.rand(B,P).to(x.device)

            batch_rand_perm = rand_num.sort(dim=1)[1]

            batch_rand_perm = batch_rand_perm[:,:s_num-P]

            batch_rand_perm = batch_rand_perm[:,None,:].repeat(1,C,1)

            select_point = torch.gather(x,2,batch_rand_perm)
            select_point = torch.cat([x,select_point],dim=2)

        if C == 5:
            _,indx = self.knn_k(x[:,:3,:].contiguous(),select_point[:,:3,:].contiguous())

        else:
            _,indx = self.knn_k(x[:,:3,:].contiguous(),select_point[:,:3,:].contiguous())

        try:
            assert(select_point.shape[2] == s_num)
        except:
            print('Error')

        # feature shape B, P1, n1, C
        feature = gether_neibour(x, indx, s_num, self.k)
        # return feature.permute(0, 3, 1, 2).contiguous()

        B, P1, n1, C = feature.shape

        feature = feature.permute([0, 3, 1, 2]).reshape([B, C, P1 * n1])
        PointSample_out = PointSample_out[:,:,:,None].repeat([1,1,1,n1]).reshape([B, C, P1 * n1])
        fea_out = self.crop_points(sampled= feature, target = PointSample_out, length= length)
        fea_out = fea_out.reshape([B, C, P1, n1])
        return fea_out.contiguous()
        # feature shape B, C, P1, n1

        
class Feature_extraction_layer_basic(Module):
    def __init__(self, neighb=8, group = 32, inc = 32, outc = 64,bias = True, dilation = 4, normalized_sampling = True):
        super(Feature_extraction_layer_basic,self).__init__()
        self.sample_layer = Dynamic_sampling_M(k=neighb, dilation = dilation)

        self.PosEncoding = nn.Sequential(nn.Conv2d(25, int(outc/2), kernel_size=1, bias=bias),
                                            nn.GroupNorm(4, int(outc/2)),
                                            # nn.LeakyReLU(negative_slope=0.1),
                                            nn.GELU(),
                                            nn.Conv2d(int(outc/2), int(outc/2), kernel_size=1, bias=bias))

        self.conv = nn.Sequential(nn.Conv2d(inc+4, int(outc/2), kernel_size=1, bias=bias))

        self.attention_mode = nn.Sequential(nn.Conv2d(outc, 32, kernel_size=1, bias=bias),
                                            nn.GroupNorm(4, 32),
                                            # nn.LeakyReLU(negative_slope=0.1),
                                            nn.GELU(),
                                            nn.Conv2d(32, 1, kernel_size=1, bias=bias))

        # self.attention_final = nn.Sequential(nn.Conv1d(outc, 256, kernel_size=1, bias=bias),
        #                                     # nn.GroupNorm(4, 32),
        #                                     # nn.LeakyReLU(negative_slope=0.1),
        #                                     nn.GELU(),
        #                                     nn.Conv1d(256, 1, kernel_size=1, bias=bias),
        #                                     nn.Sigmoid())

        self.Branch1 = nn.Sequential(nn.Conv1d(outc, outc, kernel_size=1, bias=bias),
                                            nn.GroupNorm(4, outc),
                                            # nn.LeakyReLU(negative_slope=0.1),
                                            nn.GELU(),
                                            nn.Conv1d(outc, outc, kernel_size=1, bias=bias))
    def sin_cos_encoding(self,Evt):
        Evt0 = torch.sin(Evt[:,:2,:,:] * torch.pi)
        Evt1 = torch.sin(Evt[:,:2,:,:] * torch.pi*2)
        Evt2 = torch.sin(Evt[:,:2,:,:] * torch.pi*4)
        return torch.cat([Evt0, Evt1, Evt2, Evt[:,2:,:,:]], dim = 1)


    def forward(self,x, num_points,Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        
        assert(torch.isnan(x).sum()==0)

        x1 = self.sample_layer(x, num_points, Is_temp = Is_temp, Temp_ratio = Temp_ratio)

        assert(torch.isnan(x1).sum()==0)
        # x1 shape in B C P K
        B, C, P, K = x1.shape
        pos = x1[:,:5,:,0]
        
        PC = x1[:,:4,:,0]
        PC = PC[:,:,:,None].repeat([1,1,1,K])
        PA = x1[:,:4,:,:]
        temp = ((PA[:,:3,:,:]-PC[:,:3,:,:])**2).sum(1,keepdim=True).sqrt()
        assert(torch.isnan(temp).sum()==0)
        PA = self.sin_cos_encoding(PA)
        PC = self.sin_cos_encoding(PC)
        PEmb = torch.cat([PA, PC,PA-PC,temp],dim=1)

        assert(torch.isnan(PEmb).sum()==0)

        # PEmb shape in B C P Neg
        PEmb = self.PosEncoding(PEmb)

        assert(torch.isnan(PEmb).sum()==0)
        
        X00 = self.sin_cos_encoding(x1[:,:4,:,:])
        x_ = self.conv(X00)

        feature = torch.cat([x_, PEmb],dim=1)
        
        varid = feature.std(3)

        attention = self.attention_mode(feature)
        attention = F.softmax(attention,dim = -1)

        assert(torch.isnan(attention).sum()==0)

        feature = feature * attention
        feature = feature.sum(-1,keepdim=False)

        assert(torch.isnan(feature).sum()==0)

        output = self.Branch1(feature)
        # output_aten = self.attention_final(varid)

        # output = output * output_aten

        return torch.cat([x1[:,:4,:,0], output],dim=1)


class Dynamic_emsemble(Module):
    def __init__(self,k, dilation = 1):
        super(Dynamic_emsemble, self).__init__()
        self.k = k
        self.knn_k = KNN(k=k, transpose_mode=False)

    def forward(self, Target, Source, s_num = 4, Is_temp=False, Temp_ratio=0):

        B,C,P_T = Target.shape
        B,C,P_S = Source.shape
        self.k = s_num
        indx = self.knn_k(Source[:,4:,:].contiguous(),Target[:,4:,:].contiguous())
        # output feature shape B, P, n, C
        feature = gether_neibour(Source, indx, s_num, self.k)

        # output feature shape B, C, P, N
        return feature.permute(0, 3, 1, 2).contiguous()


class Feature_extraction_layer(Module):
    def __init__(self, neighb=8, group = 32, inc = 32, outc = 64,bias = True, dilation = 4, normalized_sampling = True):
        super(Feature_extraction_layer,self).__init__()
        # if normalized_sampling:
        #     self.sample_layer = Dynamic_sampling_M(k=neighb, dilation = dilation)
        # else:
        #     self.sample_layer = Dynamic_sampling(k=neighb, dilation = dilation)

        self.sample_layer = Dynamic_emsemble(k = neighb, dilation = dilation )
        self.PosEncoding = nn.Sequential(nn.Conv2d(13, int(outc/2), kernel_size=1, bias=bias),
                                            nn.GroupNorm(4, int(outc/2)),
                                            nn.GELU(),
                                            nn.Conv2d(int(outc/2), int(outc/2), kernel_size=1, bias=bias))

        # self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=bias))

        self.conv = nn.Sequential(nn.Linear(inc, 256,  bias=bias),
                                            nn.LayerNorm(256),
                                            nn.GELU(),
                                            nn.Linear(256, outc, bias=bias))

        self.expand_conv = nn.Sequential(nn.Linear(inc, outc,  bias=bias))

        self.attention_weight = nn.Sequential(nn.Linear(outc, 128, bias=bias),
                                            nn.LayerNorm(128),
                                            nn.GELU(),
                                            nn.Linear(128, 1,  bias=bias))

        self.Branch1 = nn.Sequential(nn.Linear(outc, outc,  bias=bias),
                                            nn.LayerNorm(outc),
                                            # nn.LeakyReLU(negative_slope=0.1),
                                            nn.GELU(),
                                            nn.Linear(outc, outc,  bias=bias))
        # self.Atten_conv = nn.Sequential(nn.Linear(outc, 256,  bias=bias),
        #                                     nn.LayerNorm(256),
        #                                     # nn.LeakyReLU(negative_slope=0.1),
        #                                     nn.GELU(),
        #                                     nn.Linear(256, outc,  bias=bias),
        #                                     nn.Sigmoid())
        # self.fea_norm = nn.LayerNorm([outc])


    def forward(self,x, num_points,Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        # split_num = int(2*P/3)
        x_target = x[:,:,:]
        x_source = x[:,:,:]
    
        x1 = self.sample_layer(Target = x_target.clone(), Source = x_source, s_num = 4)
        # x_target1 shape B, P, N, c1
        x1 = x1.permute([0,2,3,1])

        # x_target1 shape B, P, N+1, c1
        x1 = torch.cat([x1, x_target.clone()[:,:,None,:].permute([0,3,2,1])], dim = 2)

        # x1 shape B, P, N, c2
        # print(' ------------- shape of x1:{}'.format(x1.shape))
        x1 = x1[:,:,:,4:]
        x1 = self.conv(x1)
        varid = x1.std(2)

        attention = self.attention_weight(x1)
        attention = F.softmax(attention,dim = 2)

        x1 = x1*attention
        x1 = x1.sum(2,keepdim=False)

        x_out = x1 + self.expand_conv(x_target.permute([0,2,1])[:,:,4:])

        # mul_attn = self.Atten_conv(varid)
        # x_out = x_out* mul_attn
        # x_out = self.fea_norm(x_out)

        return torch.cat([x_target[:,:4,:], x_out.permute([0,2,1])],dim=1)



class Feature_extraction(Module):
    def __init__(self, group=32):
        super(Feature_extraction, self).__init__()

        self.Embedding_layer01 = Feature_extraction_layer_basic(neighb=16, group = 4, inc = 4, outc = 128, dilation=1, normalized_sampling = True)

        # self.Embedding_layer02 = Feature_extraction_layer(neighb=6, inc = 64, outc = 64,)
    def forward(self,x, out_num=256, ratio_list = [0.5, 0.25, 0.16, 0.1], Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer01(x, out_num, Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # x1 = self.Embedding_layer02(x, 784, Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        return x