import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
import numpy as np
from model.mamba_4D import mamba_4D


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
     

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Patchify(nn.Module):
    def __init__(self,H_img=224,W_img=224, patch_size=3, in_chans=3, embed_dim=768):
        super(Patchify, self).__init__()
        # assert img_size % patch_size == 0
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img// patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[0], self.patch_size[1])
        P = x.shape[2]
        # print(x.shape,P)
        x = x.contiguous().view(B, self.in_chans, self.H * self.W, self.patch_size[0], self.patch_size[1])
       
        embedded_tokens = torch.einsum('ijklm,cjlm->ick', x, self.WE) + self.bE.view(1, -1, 1)
        embedded_tokens = embedded_tokens.permute(0, 2, 1).contiguous()
        embedded_tokens = self.layer_norm(embedded_tokens)
   
        return embedded_tokens    


class DePatchify(nn.Module):
    def __init__(self,H_img=224,W_img=224, patch_size=3, embed_dim=3, in_chans=768):
        super(DePatchify, self).__init__()
        # assert img_size % patch_size == 0
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img //patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x):
        
        x=self.layer_norm(x)
        y= x  + self.bE.reshape(1,1,-1)
       
        y=torch.einsum('ijk,klmn->iljmn',y,self.WE)
        y=y.reshape(y.shape[0],y.shape[1], self.H,self.W,y.shape[3], y.shape[4]).permute(0,1,2,4,3,5)
        y=y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3],y.shape[4]*y.shape[5])
        H, W = self.H , self.W

        return y

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, H=85, W=85):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.hydra = mamba_4D(d_model=dim, d_state=64, ssm_ratio=2.0)
     
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H= H
        self.W =W
        self.dim=dim
    
    def forward(self, x):
        inp = x
        outp  = self.drop_path(self.hydra(self.norm1(x).reshape(-1, self.H, self.W, self.dim)).reshape(-1, self.H*self.W, self.dim))
        x = inp + outp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, inp, outp


class Model(nn.Module):
    def __init__(self,
                 space_dim=2,
                 fun_dim=1,
                 out_chans=1,
                 H_img=85,  #actual input dimension
                 W_img=85,
                 embed_dims=[256,256,256],
                 num_heads=[8,8,8],
                 mlp_ratios=[4,4,4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[6,6,6],
                 patch_sizes=[2,4,8],
                 H=88,  #input dimension after padding
                 W=88,
                 num_scales=3,
                 get_last_output=False):
        super().__init__()
        for i in range(num_scales):
            patch_embed = Patchify(  H_img=H,
                                     W_img=W,
                                     patch_size=patch_sizes[i],
                                     in_chans=fun_dim+space_dim if i==0 else embed_dims[i-1],
                                     embed_dim=embed_dims[i])
            num_patches = (H//patch_sizes[i])*(W//patch_sizes[i])
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0
            
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, H = H//patch_sizes[i], W = W//patch_sizes[i] )
                for j in range(depths[i])])
            cur += depths[i]
            depatch_embed= DePatchify(  H_img=H,
                                        W_img=W,
                                        patch_size=patch_sizes[i],
                                        embed_dim=embed_dims[i],
                                        in_chans=out_chans if i==num_scales-1 else embed_dims[i])
             
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
            setattr(self,f"depatch_embed{i+1}",depatch_embed)
            self.num_scales=num_scales
            self.get_last_output=get_last_output
            self.H=H
            self.W=W
            self.H_img=H_img
            self.W_img=W_img
            self.space_dim=space_dim
            self.fun_dim=fun_dim

            self.placeholder = nn.Parameter((1 / (space_dim)) * torch.rand(space_dim, dtype=torch.float))
   
            
    def forward_features(self, x):
        outs = []
        features = []
        B = x.shape[0]
        for i in range(self.num_scales):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            depatch_embed= getattr(self, f"depatch_embed{i+1}")

            x = patch_embed(x)
            x = pos_drop(x + pos_embed)
            
            for blk in block:
                x, inp, outp = blk(x)
                features.append(inp)
                features.append(outp)

            x=depatch_embed(x)
            outs.append(x)
        return outs, features

    def forward(self,x,fx):
        x=x.reshape(-1,self.H_img,self.W_img,self.space_dim)
        
        if fx is not None:
            fx=fx.reshape(-1,self.H_img,self.W_img,self.fun_dim)
            fx = torch.cat((x, fx), -1)
        else:
            fx = x + self.placeholder[None, None, :]
         
        fx=fx.permute(0,3,1,2)
      
        pad_h = self.H-self.H_img  
        pad_w = self.W-self.W_img 
        if pad_h > 0 or pad_w > 0:
            fx = F.pad(fx, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        fx, features = self.forward_features(fx)
        fx=fx[-1]
        
        if pad_h > 0 or pad_w > 0:
              fx = fx[:, :, :self.H_img, :self.W_img]
        fx=fx.reshape(fx.shape[0],-1)
        return fx
     
    def get_grid(self, batchsize=4):
        size_x, size_y = self.H_img, self.W_img
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)
        return grid      

