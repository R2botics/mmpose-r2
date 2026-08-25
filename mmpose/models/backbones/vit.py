"""ViTPose's plain-ViT backbone, ported to the mmpose 1.x registry.

Ported from ViTPose (NeurIPS 2022) `mmpose/models/backbones/vit.py`. Three
deliberate deviations from that source, each load-bearing here:

1. **No timm dependency.** `drop_path`/`to_2tuple`/`trunc_normal_` come from
   mmcv/mmengine/torch instead, which this environment already has.

2. **Transformer blocks are named `layers`, not `blocks`.**
   `mmpose.engine.optim_wrappers.LayerDecayOptimWrapperConstructor` assigns a
   layer id by string-matching `backbone.layers.<i>`; anything it does not
   recognise falls into its `else` branch and receives `lr_scale = 1.0`. Keep
   the upstream `blocks` name and layer-wise LR decay silently becomes a no-op
   for every transformer block -- no error, just a config that does not do what
   it says. `scripts/convert_vit_to_rgbd.py` renames the checkpoint keys to
   match.

3. **Extra input channels live in a SEPARATE stem** (`self.extra_stem`), not in
   widened `patch_embed.proj` weights. See `ExtraChannelStem` below for why.

Everything else -- attention, MLP, the `ratio` sub-patch stride trick, the
frozen-stage plumbing -- is upstream behaviour.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks.drop import DropPath
from mmengine.utils import to_2tuple
from torch.nn.init import trunc_normal_

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding.

    `ratio` > 1 shrinks the CONV STRIDE while keeping the 16x16 kernel, giving
    overlapping patches on a finer grid: ratio=2 turns a 256x256 input from a
    16x16 token grid (stride 16) into 32x32 (stride 8). Attention cost grows
    with the square of the token count, so ratio=2 is roughly 16x the attention
    FLOPs -- but it is the only knob that makes a plain ViT operate below
    stride 16, which matters when the structure you need to resolve is smaller
    than one patch.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]) * (ratio**2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio),
                            int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]),
                                   int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ExtraChannelStem(nn.Module):
    """Tokenizer for input channels 3..N-1 (depth, validity mask, ...).

    WHY A SEPARATE STEM RATHER THAN A WIDENED ``patch_embed.proj``

    A ViT's patch embed is not a first conv layer that a following BatchNorm
    will re-standardise -- it IS the tokenizer, and the 12 pretrained blocks
    downstream were fitted to the token distribution it produces. Whatever the
    extra channels contribute lands directly in that distribution.

    The mean-inflation trick used for HRNet (`scripts/convert_hrnet_to_rgbd.py`
    copies mean(RGB) into the new channels) is the wrong default here for two
    reasons. It assumes the new channels are photometrically like RGB, which a
    height field and a binary validity mask are not; and HRNet survives it only
    because `conv1` is immediately followed by a BatchNorm that re-standardises
    whatever comes out. A ViT has no normalisation between the patch projection
    and the first residual stream -- the first LayerNorm sits INSIDE block 0,
    after `x + pos_embed`, and it rescales each token's norm but cannot undo a
    change of DIRECTION in the 768-d embedding space. Mean-inflation moves every
    token off the manifold the pretrained blocks expect, before a single
    gradient step.

    So the default here is a zero-initialised parallel projection whose output
    is ADDED to the RGB tokens:

        token = proj_rgb(x[:, :3]) + proj_extra(x[:, 3:]),   proj_extra := 0

    At step 0 this is BIT-IDENTICAL to the pretrained RGB model, and the extra
    channels still receive gradient from step 1 (d L/d W_extra = dL/dtoken (x)
    x_extra, which is nonzero wherever the depth input is). Nothing is dead.

    Mathematically, zero-init'ing the extra slice of a single widened conv is
    the SAME function -- this is a re-parameterisation, not a different model.
    It is factored out for three practical reasons:

      * **Learning rate.** `LayerDecayOptimWrapperConstructor` maps
        `backbone.patch_embed*` to layer 0, whose scale is
        `layer_decay_rate ** (num_layers + 1)` -- 0.024x base LR at the usual
        0.75/12. Zero-initialised weights at 2.4% of base LR barely leave zero,
        so depth would contribute nothing and the ablation would read "depth
        does not help" when what it measured was "depth never trained". Because
        this module is named `extra_stem` and NOT `patch_embed*`, that
        constructor's `else` branch gives it `lr_scale = 1.0`. The naming is
        load-bearing; do not rename it into the patch_embed namespace.
      * **A real ablation.** RGB and RGB+D runs then share one checkpoint and
        one RGB stem. The only difference is whether this module exists, so a
        difference in the metrics is attributable to depth rather than to two
        differently-initialised tokenizers.
      * **Freezing.** The RGB stem can be frozen through synthetic pretraining
        while the depth stem trains, without splitting a weight tensor.

    Set ``mode='inflate_mean'`` to reproduce the HRNet-style behaviour instead
    (the extra channels are initialised from the mean RGB filter and scaled by
    ``1 / n_extra`` so the token norm is preserved rather than inflated). It is
    one config line, so the claim above is testable rather than assumed.

    Args:
        n_extra (int): Number of channels beyond RGB.
        embed_dim (int): Token dimension.
        patch_size (int): Conv kernel size, matching PatchEmbed.
        stride (int): Conv stride, matching PatchEmbed.
        padding (int): Conv padding, matching PatchEmbed.
        mode (str): ``'zero'`` (default) or ``'inflate_mean'``.
    """

    VALID_MODES = ('zero', 'inflate_mean')

    def __init__(self, n_extra, embed_dim, patch_size, stride, padding,
                 mode='zero'):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f'extra_stem mode must be one of {self.VALID_MODES}, got {mode!r}')
        self.n_extra = n_extra
        self.mode = mode
        # bias=False: `patch_embed.proj` already contributes the token bias.
        # A second bias here would be added on top of it, shifting every token
        # by a learnable constant that has nothing to do with the depth input.
        self.proj = nn.Conv2d(
            n_extra, embed_dim, kernel_size=patch_size, stride=stride,
            padding=padding, bias=False)
        nn.init.zeros_(self.proj.weight)

    def init_from_rgb(self, rgb_weight):
        """Seed from the pretrained RGB filters, for ``mode='inflate_mean'``.

        Called after the pretrained checkpoint is loaded, so it overwrites the
        zeros with mean(RGB). Scaled by 1/n_extra: copying the mean filter
        n_extra times ADDS n_extra times the mean-channel response on top of
        the RGB response, and while the first LayerNorm rescales each token it
        does so AFTER the extra channels have already rotated the token
        direction. Dividing keeps the expected contribution at the scale of one
        RGB channel.

        Args:
            rgb_weight (Tensor): ``patch_embed.proj.weight``, (embed, 3, k, k).
        """
        if self.mode != 'inflate_mean':
            return
        with torch.no_grad():
            mean_filter = rgb_weight.mean(dim=1, keepdim=True)
            self.proj.weight.copy_(
                mean_filter.repeat(1, self.n_extra, 1, 1) / self.n_extra)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


@MODELS.register_module()
class ViT(BaseBackbone):
    """Plain Vision Transformer backbone for ViTPose, with N-channel input.

    Args:
        img_size (int | tuple): Input resolution. Defaults to 256.
        patch_size (int): Patch kernel size. Defaults to 16.
        in_channels (int): Number of input channels. 3 = RGB, 4 = RGB+height,
            5 = RGB+height+validity mask. Anything above 3 is handled by
            :class:`ExtraChannelStem`. Defaults to 3.
        embed_dim (int): Token dimension. Defaults to 768 (ViT-B).
        depth (int): Number of transformer blocks. Defaults to 12 (ViT-B).
        num_heads (int): Attention heads. Defaults to 12 (ViT-B).
        ratio (int): Sub-patch stride factor; see :class:`PatchEmbed`.
            Defaults to 1 (stride 16).
        extra_stem_mode (str): ``'zero'`` or ``'inflate_mean'``; see
            :class:`ExtraChannelStem`. Defaults to ``'zero'``.
        freeze_rgb_stem (bool): Freeze `patch_embed` while leaving
            `extra_stem` trainable. Useful when adapting an RGB-pretrained
            model to depth without disturbing the RGB tokenizer.
            Defaults to False.
        last_norm (bool): Apply a final LayerNorm. Defaults to True.
        use_checkpoint (bool): Gradient checkpointing. Defaults to False.
        frozen_stages (int): Freeze `patch_embed` and blocks 1..frozen_stages.
            -1 freezes nothing. Defaults to -1.
        freeze_attn (bool), freeze_ffn (bool): Partial freezing, as upstream.
        drop_rate, attn_drop_rate, drop_path_rate (float): Regularisation.
        qkv_bias (bool), qk_scale (float), mlp_ratio (float), norm_layer:
            Standard ViT knobs.
    """

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 ratio=1,
                 last_norm=True,
                 extra_stem_mode='zero',
                 freeze_rgb_stem=False,
                 freeze_attn=False,
                 freeze_ffn=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if in_channels < 3:
            raise ValueError(
                f'in_channels must be >= 3 (RGB is the pretrained stem); '
                f'got {in_channels}')

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.freeze_rgb_stem = freeze_rgb_stem
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        # RGB tokenizer, identical in shape to the pretrained one.
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # Channels 3.. go through their own zero-initialised projection.
        # NOTE the attribute name: it must NOT start with `patch_embed`, or
        # LayerDecayOptimWrapperConstructor drops it to layer 0 (~2.4% of base
        # LR) and a zero-init stem never leaves zero. See ExtraChannelStem.
        self.extra_stem = None
        if in_channels > 3:
            self.extra_stem = ExtraChannelStem(
                n_extra=in_channels - 3,
                embed_dim=embed_dim,
                patch_size=self.patch_embed.proj.kernel_size,
                stride=self.patch_embed.proj.stride,
                padding=self.patch_embed.proj.padding,
                mode=extra_stem_mode)

        # +1 for the pretrained checkpoint's class token, which ViTPose keeps
        # in the parameter but folds into every token in forward().
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # `layers`, not `blocks` -- see the module docstring.
        self.layers = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self._freeze_stages()

    def init_weights(self):
        """Initialise weights, then apply the extra-stem seeding.

        Order matters for ``extra_stem_mode='inflate_mean'``: the mean RGB
        filter can only be read after the pretrained checkpoint has landed in
        ``patch_embed.proj.weight``, so seeding runs AFTER super().
        """
        if self.init_cfg is None:
            def _init(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            self.apply(_init)
        else:
            super().init_weights()

        if self.extra_stem is not None:
            self.extra_stem.init_from_rgb(self.patch_embed.proj.weight)

    def _freeze_stages(self):
        if self.frozen_stages >= 0 or self.freeze_rgb_stem:
            # Only the RGB tokenizer. `extra_stem` is deliberately left
            # trainable: freezing a zero-initialised stem would pin the depth
            # channels to "contribute nothing" for the whole run.
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(self.depth):
                m = self.layers[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(self.depth):
                m = self.layers[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f'ViT was built for in_channels={self.in_channels} but got a '
                f'{C}-channel input. Check the data pipeline and the data '
                f'preprocessor mean/std length.')

        tokens, (Hp, Wp) = self.patch_embed(x[:, :3])
        if self.extra_stem is not None:
            tokens = tokens + self.extra_stem(x[:, 3:])

        # ViTPose folds the class-token position into every patch token rather
        # than carrying a separate token; pos_embed[:, :1] is a constant offset.
        tokens = tokens + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for layer in self.layers:
            if self.use_checkpoint:
                tokens = checkpoint.checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)

        tokens = self.last_norm(tokens)
        feat = tokens.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        # mmpose 1.x heads index `feats[-1]`, so return a tuple.
        return (feat, )

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
