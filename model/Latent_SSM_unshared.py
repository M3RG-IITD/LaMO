import torch.nn as nn
import torch
from einops import rearrange, repeat
from model.mamba_2D_v2 import Hydra
from model.mamba_2D_v1 import Mamba2

class Latent_SSM_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.slice_num= slice_num
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.hydra = Hydra(
                d_model=inner_dim, # Model dimension d_model
                d_state=64,  # SSM state expansion factor
                d_conv=3,    # Local non-causal convolution width
                expand=1,    # Block expansion factor
                use_mem_eff_path=False,    # Nightly release. Thanks to Alston Lo
            )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        
        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G

        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        #ssm on slice tokens
        slice_token = rearrange(slice_token, 'b h g c -> b g (h c)')
        out_slice_token = self.hydra(slice_token)
        out_slice_token = rearrange(out_slice_token, 'b g (h c) -> b h g c ', h=self.heads, c=self.dim_head)

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Latent_SSM_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.slice_num =slice_num
         
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
                
        self.hydra = Mamba2(   #bimamba2
                d_model=inner_dim, # Model dimension d_model
                d_state=64,  # SSM state expansion factor
                d_conv=3,    # Local non-causal convolution width
                expand=2,    # Block expansion factor
                headdim=32,
            )
      
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid)/ torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
      
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
    
        inp=slice_token      
        slice_token = rearrange(slice_token, 'b h g c -> b g (h c)')
        out_slice_token = self.hydra(slice_token)
        out_slice_token = rearrange(out_slice_token, 'b g (h c) -> b h g c ', h=self.heads, c=self.dim_head)
        outp=out_slice_token
        
        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x), inp, outp
         

