import sys
import os
sys.path.append('../')

import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn
    
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    

class mamba_4D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16, # now with mamba2, dstate should be bigger...
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        # ======================
        forward_type="m0",
        # ======================
        with_initial_state=False,
        # ======================
        # ==============================
        force_fp32=False, # True: input fp32
        chunk_size = 64,
        # ==============================
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        # assert ===================
        assert d_inner % dt_rank == 0
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        
        self.with_dconv = d_conv > 1
        self.d_state = d_state
        self.force_fp32 = force_fp32
        self.chunk_size = chunk_size
        Linear = nn.Linear
        self.forward = self.forward

        self.disable_force32 = False
        self.oact = False
        self.disable_z = False
        self.disable_z_act = False
        self.out_norm = nn.LayerNorm(d_inner)
        
        self.selective_scan_backend = selective_scan_backend
        self.scan_mode = scan_mode
        self.scan_force_torch = scan_force_torch

        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            ) 
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
        
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

    def forward_core(
        self,
        x: torch.Tensor=None, 
        **kwargs,
    ):
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = self.d_state
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        
        assert RD == R * D
        
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[self.scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == self.d_state
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch) # (B, H, W, 4, D)
        
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        
        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        
        ys, final_state = mamba_chunk_scan_combined(
            xs, dts, As, Bs, Cs, chunk_size = self.chunk_size, D=Ds, dt_bias=dt_bias, 
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
        )
                
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch)
            
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):

        x = self.in_proj(x)
        
        if not self.disable_z:
            x, z = x.chunk(2, dim=(-1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out
