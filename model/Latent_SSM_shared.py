import torch.nn as nn
import torch
from einops import rearrange, repeat
from model.mamba_2D_v2 import Hydra

class Latent_SSM_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.hydra = Hydra(
                d_model=inner_dim, # Model dimension d_model
                d_state=64,  # SSM state expansion factor
                d_conv=3,    # Local non-causal convolution width
                expand=2,    # Block expansion factor
                use_mem_eff_path=False,    # Nightly release. Thanks to Alston Lo
            )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, slice_token, slice_weights):
        
        #Hydra on slice tokens
        slice_token = rearrange(slice_token, 'b h g c -> b g (h c)')
        out_slice_token = self.hydra(slice_token)
        out_slice_token = rearrange(out_slice_token, 'b g (h c) -> b h g c ', h=self.heads, c=self.dim_head)
         
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
        self.H = H
        self.W = W

        self.hydra = Hydra(
                d_model=inner_dim, # Model dimension d_model
                d_state=1,  # SSM state expansion factor
                d_conv=3,    # Local non-causal convolution width
                expand=1,    # Block expansion factor
                use_mem_eff_path=True  # Nightly release. Thanks to Alston Lo
            )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, slice_token, slice_weights):
     
        #Hydra on slice tokens
        inp=slice_token      

        slice_token = rearrange(slice_token, 'b h g c -> b g (h c)')
        out_slice_token = self.hydra(slice_token)
        out_slice_token = rearrange(out_slice_token, 'b g (h c) -> b h g c ', h=self.heads, c=self.dim_head)
        outp=out_slice_token
                
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x), inp, outp

