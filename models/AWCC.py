import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange
from models.transformer_decoder import Block_dec
from torch import Tensor
from torch.nn import functional as F

model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}
cfg = {'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

##################################################################################
# CC
##################################################################################


class AWCCNet(nn.Module):

    def __init__(self, use_pe=False, output_dim=1):
        super().__init__()
        # ===== parmas =====
        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "gelu"
        
        self.n_prototypes = n_prototypes = 8
        self.n_tokens = n_tokens = 48
        # ===== parmas =====

        self.features = make_layers(cfg['E'])

        self.weather_encoder = WeatherEncoder(n_downsample=4, dim=64, style_dim=n_prototypes, norm='none', activ='relu', pad_type='reflect', input_dim=512)
        self.weather_bank = nn.Parameter(torch.randn((n_prototypes, n_tokens, d_model)), requires_grad=True)

        self.pos_encoder = PositionalEncoding2d(d_model) if use_pe else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=None)

        depths = [3, 4, 6, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        self.decoder = nn.ModuleList([Block_dec(
                                    dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[i], 
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                    for i in range(depths[0])])
        
        self.reg_decoder = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
                                         nn.Conv2d(128, output_dim, 1)
                                        )

        # --- load pretrain ---
        vgg_pretrained_checkpoint = model_zoo.load_url(model_urls['vgg19'])
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(vgg_pretrained_checkpoint, prefix='features.')
        msg = self.features.load_state_dict(vgg_pretrained_checkpoint, strict=False)
        print('-' * 40)

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        
        vgg_feats = self.features(x)

        self.weight_vectors = self.weather_encoder(vgg_feats, return_logit=True)
        self.weight_vectors = F.softmax(self.weight_vectors, dim=-1)

        # transformer encoder
        bs, c, h, w = vgg_feats.shape
        vgg_feats = vgg_feats.flatten(2).permute(0, 2, 1)  # [b, h*w, c] = [b, n_tokens, dim]
        vgg_feats = vgg_feats[:1]
        if self.pos_encoder is not None:
            vgg_feats = self.pos_encoder(vgg_feats, h, w, nbc=False)
        x = self.encoder(vgg_feats)  # transformer [b, n_tokens, dim]

        # transformer decoder
        style_q = torch.einsum('bs, snc -> bnc', self.weight_vectors, self.weather_bank)
        for blk in self.decoder:
            x = blk(x, h, w, style_q[:1])
            
        # regression head
        x = x.permute(0, 2, 1).view(1, c, h, w)
        x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=True)
        x = self.reg_decoder(x)
        return x, style_q.flatten(1)

        
class PositionalEncoding2d(nn.Module):
    # https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

    def __init__(self, d_model: int, height: int = 16, width: int = 16):
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(1, d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0, 0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[0, 1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[0, d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[0, d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, h, w, nbc=True) -> Tensor:
        """
        Args:
            x: Tensor, shape [n, b, c]
        """
        if h != 16 or w != 16:
            pe = F.interpolate(self.pe, size=(h, w), mode='bicubic', align_corners=False)
        else:
            pe = self.pe

        if nbc:
            pe = rearrange(pe, 'b c h w -> (h w) b c')
        else:
            pe = rearrange(pe, 'b c h w -> b (h w) c')
            
        x = x + pe
        return x

class WeatherEncoder(nn.Module):
    def __init__(self, n_downsample, dim, style_dim, norm, activ, pad_type, input_dim=3):
        super().__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling

        self.model = nn.Sequential(*self.model)

        self.fc = nn.Conv2d(dim, style_dim, 1, 1, 0)

        self.output_dim = dim

    def forward(self, x, return_logit=False):
        feats = self.model(x)
        if return_logit:
            logits = self.fc(feats).view(x.size(0), -1) # 8d
            return logits
        else:
            return self.fc(feats)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19_trans(use_pe=False):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """

    return AWCCNet(use_pe)
        
