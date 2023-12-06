import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np
from PIL import Image

from .backbone import build_backbone
from .vit_utils import  DropPath, trunc_normal_
from einops import rearrange, reduce, repeat

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)
    
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x),y

class ECABasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = x
        out, weight = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, weight

class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _ = x.size()

        # Style pooling
        # AvgPool（全局平均池化）：
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        # StdPool（全局标准池化）
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        # CFC（全连接层）
        z = self.cfc(u)  # (b, c, 1)
        # BN（归一化）
        z = self.bn(z)
        # Sigmoid
        g = torch.sigmoid(z)


        g = g.view(b, c, 1)
        return x * g.expand_as(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.q = nn.Linear(dim, dim, bias=qkv_bias)
           self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)


    def forward(self, qry,tgt):
        B, Nq, C = qry.shape
        B, N, C = tgt.shape
        if self.with_qkv:
           qry = self.q(qry).reshape(B, Nq,self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           tgt = self.kv(tgt).reshape(B, N,2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           key,value = tgt[0],tgt[1]
        else:
           qry = qry.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           key, value  = tgt

        attn = (qry @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(B, Nq, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x, attn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm,num_frames = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.tam = nn.Sequential(
            nn.Conv1d(dim,
                      dim // 4,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm1d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid())
        

    def forward(self, x, T):
        B = x.shape[0] // T
        res_s, attn = self.attn(self.norm1(x))
        res_s = self.drop_path(res_s)
        xs = x + res_s

        xsp = xs[:,1:,:]
        xt = F.adaptive_avg_pool1d(rearrange(xsp,'(b t) n d -> (b d) t n',t =T), 1).squeeze(2)
        xt = rearrange(xt,'(b d) t -> b d t',b =B)
        xt_weight = self.tam(xt).unsqueeze(3)
        xt = rearrange(xsp,'(b t) n d -> b d t n',t =T) * xt_weight
        cls_token = torch.mean(xt,dim =3).unsqueeze(3)
        xt = torch.cat((cls_token,xt),dim = 3)
        x =  rearrange(xt,'b d t n -> (b t) n d')
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class Decoder(nn.Module):
    def __init__ (self, depth, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., dropout=0., with_qkv=True):
        super().__init__()
        self.depth = depth
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.decoder = Cross_Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, with_qkv) 
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=dropout)
    def forward(self,local_info,gloable_info):
        for i in range(self.depth):
            local_feature,_= self.decoder(local_info,gloable_info)
            local_feature = self.norm1(local_feature)
            local_feature = local_info + local_feature
            local_info = local_info + self.norm3(self.mlp(self.norm2(local_feature)))
        return local_info

class weight_alter(nn.Module):
    def __init__(self,method,embed_dim):
        super().__init__()
        self.method = method
        self.embed_dim = embed_dim
    
    def get_indicator(self, scores, k, sigma=0.1):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def forward(self, input,src,weight_attn,k2):
        if self.method == 'three_stage':
            bbox =[]
            weight_attn = weight_attn.reshape(weight_attn.shape[0],-1)
            self.num_samples = weight_attn[0].shape[0]
            indicators = self.get_indicator(weight_attn, k2)
            indicators = rearrange(indicators, "b d k -> b k d")
            sorted_index = torch.einsum("b k d, b d c -> b k c",
                         indicators, weight_attn.unsqueeze(-1))
            
            for l in range(weight_attn.shape[0]):
                for index in range(k2):
                    # bbox0 = []
                    if sorted_index[l][index]//40 < 1:
                        if sorted_index[l][index]%40 < 1:
                            bbox.append(sorted_index[l][index]%40*32)
                            bbox.append(sorted_index[l][index]//40*32)
                            bbox.append((sorted_index[l][index]%40+2)*32)
                            bbox.append((sorted_index[l][index]//40+2)*32)
                        else:
                            bbox.append((sorted_index[l][index]%40-1)*32)
                            bbox.append(sorted_index[l][index]//40*32)
                            bbox.append((sorted_index[l][index]%40+1)*32)
                            bbox.append((sorted_index[l][index]//40+2)*32)
                    else:
                        if sorted_index[l][index]%40 < 1:
                            bbox.append(sorted_index[l][index]%40*32)
                            bbox.append((sorted_index[l][index]//40-1)*32)
                            bbox.append((sorted_index[l][index]%40+2)*32)
                            bbox.append((sorted_index[l][index]//40+2)*32)
                        else:
                            bbox.append((sorted_index[l][index]%40-1)*32)
                            bbox.append((sorted_index[l][index]//40-1)*32)
                            bbox.append((sorted_index[l][index]%40+1)*32)
                            bbox.append((sorted_index[l][index]//40+2)*32)
            return bbox
        elif self.method == 'group1':
            weight_attn1 = torch.zeros(weight_attn.shape[0],21,40)
            bbox =[]
            for s in range(weight_attn.shape[0]):
                for i in range(0,21,1):
                    for j in range(40):
                        weight_attn1[s,i,j]=weight_attn[s][i][j] + weight_attn[s][i+1][j] +weight_attn[s][i+2][j]
                sorted_index = self.get_indicator(weight_attn1[s].reshape(-1), k2)
                for index in range(k2):
                    bbox.append((sorted_index[index]%40)*32)
                    bbox.append((sorted_index[index]//40)*32)
                    bbox.append((sorted_index[index]%40+1)*32)
                    bbox.append((sorted_index[index]//40+3)*32)
            return bbox

class Res_Patch(nn.Module):
    def __init__(self, args):
        super(Res_Patch, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities
        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048
        self.embed_dim=512
        self.num_patches=196
        self.depth = args.depth
        self.in_chans = 3
        self.dropout = nn.Dropout(args.dropout)
        self.drop_path_rate=0.2
        self.drop_rate=0.2
        self.top_k2 = args.top_k2

        # feature extraction
        self.backbone = build_backbone(args)
        self.backbone2 = build_backbone(args)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 921, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                drop=self.drop_rate, attn_drop=self.drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm,num_frames = self.num_frames)
            for i in range(2)])
        self.spatial_norm = nn.BatchNorm1d(self.embed_dim)

        #key_patch_select
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.weight_decide = weight_alter('three_stage',self.embed_dim)

        # # channel
        self.channel_blocks = nn.ModuleList([
            SRMLayer(self.embed_dim)
            for i in range(1)])
    
        self.decoder = Decoder(2 ,self.embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=False, 
                qk_scale=None, attn_drop=self.drop_rate, proj_drop=self.drop_rate, dropout = self.drop_rate, with_qkv=True)
        self.fusion_diff = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU()
            )

        self.fusion=nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        

        #spatial_fusion
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(2*self.embed_dim, self.num_class)

        self.relu = F.relu
        self.gelu = F.gelu

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed','cls_token'}

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h0, w0 = x.shape
        x1 = x.reshape(b * t, 3, h0, w0)

        
        src_1, _ = self.backbone(x1)  # [B x T, C, H/32, W/32]
        _, d, h, w = src_1.shape
        src_1 = rearrange(src_1,'bt d h w -> bt (h w) d')
        src = src_1

        #spatial-tem
        src_s = torch.cat((self.cls_token.expand(src.shape[0],-1,-1),src),dim=1)
        src = src_s +self.pos_embed
        src = self.pos_drop(src)
        src_s= src
        attn_weight = []
        for blk in self.spatial_blocks:
            src_s,attn = blk(src_s,t)
            attn_weight.append(attn)
        attn_weight = torch.stack(attn_weight)        # 12 * B * H * N * N
        attn_weight = torch.mean(attn_weight, dim=2)  # 12 * B * N * N
        weight_attn = attn_weight[1,:, 0, 1:].reshape([b*t, h,w])

        src_s = rearrange(src_s, 'bt hw d -> bt d hw')
        src_s = self.spatial_norm(src_s)
        src_s = rearrange(src_s, 'bt d hw -> bt hw d')
        cls_token = src_s[:,0]
        src_s = src_s[:,1:]
        src = src_s

        #locate
        bbox = torch.tensor(self.weight_decide(src,src,weight_attn,self.top_k2)).reshape(b*t,self.top_k2,-1).detach().tolist()
        x= x.cpu().numpy()
        x=x.reshape(b*t,h0,w0,3)
        region = torch.zeros(b*t,self.top_k2,3,32,32)
        for i in range(x.shape[0]):
            img = Image.fromarray(x[i].astype('uint8'))
            for j in range(self.top_k2):
                x1=img.crop((bbox[i][j][0],bbox[i][j][1],bbox[i][j][2],bbox[i][j][3]))
                x1=self.transform(x1)
                region[i][j]=x1
        region = region.cuda().reshape(b*t*self.top_k2,3,32,32)
        src_l,_ = self.backbone2(region)
        src_l =  rearrange(src_l,'(bt k) d h w -> bt k d (h w)',k =self.top_k2).squeeze(3)


        # ''' LIS '''
        src_c = rearrange(src_l, 'bt k d -> bt d k')
        for blk in self.channel_blocks:
            src_c = blk(src_c)
        src_l = rearrange(src_c, 'bt d k -> bt k d')

        # # #diff_fusion
        src_diff = rearrange(src_l,'bt k d -> bt (k d)').unsqueeze(1).repeat(1,self.top_k2,1) #bt k kd
        src_d = src_l.repeat(1,1,self.top_k2) #bt k kd
        diff = abs(src_d - src_diff).reshape(b*t,self.top_k2,self.top_k2,-1).sum(2).reshape(b*t,self.top_k2,-1)

        src_l = src_l + diff

        src_l = self.fusion_diff(src_l.reshape(b*t*self.top_k2,-1)).reshape(b*t,self.top_k2,-1)

        # #Refine
        de = self.decoder(src_l,cls_token.unsqueeze(1))

        de = self.fusion(torch.cat((cls_token.unsqueeze(1),de),dim = 1).reshape(b*t,-1,(1+self.top_k2)))

        de = self.avg_pool(de)


        representations = torch.cat((cls_token.unsqueeze(2),de),dim = 1).reshape(b*t,-1)
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)
        return activities_scores