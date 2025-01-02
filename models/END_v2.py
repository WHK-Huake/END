import torch
import torch.nn as nn
import torch.nn.functional as TF
from models.common import Conv, CAB
import numpy as np
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, m_head=0):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.m_head = m_head
        # self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1,
                                               self.n_heads).transpose(1, 2).transpose(0, 1))

    def forward(self, x):
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class TransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim=16, window_size=8, drop_path=0.0):
        """ SwinTransformer Block
        """
        super(TransBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        x = Rearrange('b h w c -> b c h w')(x)
        return x


class Conv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv_Relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DeConv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConv_Relu, self).__init__()
        self.deconv = Conv_Relu(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = TF.upsample(x, scale_factor=2)
        return self.deconv(x)


## Residual Channel Attention Network (RCAN)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks, bias=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        self.conv1 = Conv(in_feat, out_feat, 3)
        self.cab = nn.Sequential(*[CAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act) for _ in range(n_blocks)])
        self.conv2 = Conv(out_feat, out_feat, 3)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.cab(x) + x
        x = self.conv2(x1)
        return x


class PCM(nn.Module):
    def __init__(self, channels, n_feat, kernel_size, bias):
        super(PCM, self).__init__()
        self.conv1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = Conv(n_feat, channels, kernel_size, bias=bias)
        self.conv3 = Conv(channels, n_feat, kernel_size, bias=bias)
        self.conv4 = Conv(n_feat, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = self.conv3(img)
        x3 = torch.sigmoid(self.conv4(x1))
        x2 = x3 * x2
        x1 = x + x2
        return x1, img


class Priormer(nn.Module):
    def __init__(self, channel, n_feats):
        super(Priormer, self).__init__()
        self.enc1 = nn.Sequential(Conv(channel, n_feats, 3), Conv(n_feats, n_feats, 3))
        self.former1 = TransBlock(n_feats, n_feats)
        self.former2 = TransBlock(n_feats, n_feats)
        self.former3 = TransBlock(n_feats, n_feats)
        self.former4 = TransBlock(n_feats, n_feats)
        self.former5 = TransBlock(n_feats, n_feats)
        self.enc2 = Conv(n_feats, n_feats, 3)
        self.out2 = PCM(channel, n_feats, 3, bias=False)

    def forward(self, x):
        f = self.enc1(x)
        f = self.former1(f)
        f = self.former2(f)
        f = self.former3(f)
        f = self.former4(f)
        f = self.former5(f)
        f = self.enc2(f)
        feats, out = self.out2(f,x)
        return out, feats


class PGB(nn.Module):
    def __init__(self, in_channel=3, f_channel=64, g_channel=1):
        super(PGB, self).__init__()
        self.conv1 = Conv(g_channel, 1, 1)
        self.conv2 = Conv(g_channel, f_channel, 1)
        self.conv3 = Conv(in_channel, f_channel, 3)
        self.conv4 = Conv(f_channel, f_channel, 3)
        self.conv5 = Conv(f_channel, f_channel, 3)

    def forward(self, img, guide_f):
        guide_mul = torch.sigmoid(self.conv1(guide_f))
        guide_add = self.conv2(guide_f)
        x = self.conv3(img)
        x = x * guide_mul
        x = self.conv4(x)
        x = x + guide_add
        x = self.conv5(x)
        return x


class IEN(nn.Module):
    def __init__(self, in_channel=3, f_channel=48, w_channel=48):
        super(IEN, self).__init__()

        self.layer0 = nn.Sequential(
                      Conv(in_channel, f_channel, 3),
                      Conv(f_channel, f_channel, 3)
        )

        self.para = torch.nn.Parameter(torch.ones(w_channel, 1, 1))

        self.guide1 = PGB(f_channel, f_channel, w_channel)
        self.layer1 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide2 = PGB(f_channel, f_channel, w_channel)
        self.layer2 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide3 = PGB(f_channel, f_channel, w_channel)
        self.layer3 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide4 = PGB(f_channel, f_channel, w_channel)
        self.layer4 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide5 = PGB(f_channel, f_channel, w_channel)
        self.layer5 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide6 = PGB(f_channel, f_channel, w_channel)
        self.layer6 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.out = Conv(f_channel, 3, 3)

    def forward(self, img, illu, rest):
        x = self.layer0(img)
        res_illu = illu-self.para*rest
        x = self.guide1(x, res_illu)
        x = self.layer1(x)
        x = self.guide2(x, rest)
        x = self.layer2(x)
        x = self.guide3(x, res_illu)
        x = self.layer3(x)
        x = self.guide4(x, rest)
        x = self.layer4(x)
        x = self.guide5(x, res_illu)
        x = self.layer5(x)
        x = self.guide6(x, rest)
        x = self.layer6(x)
        x = self.out(x)

        return x


### Add instance norm in illum module
class Illum_YCRCB_Denoise_IN(nn.Module):
    def __init__(self):
        super(Illum_YCRCB_Denoise_IN, self).__init__()

        self.illu = Priormer(channel=1, n_feats=64)               # L-PEN
        self.denoise = Priormer(channel=2, n_feats=64)            # C-PEN
        self.FuN = IEN(3, 64, 64)

    def forward(self, x):
        y = x[:, 0, :, :].unsqueeze(1)
        uv = x[:, 1:3, :, :]

        out_y, out_y_feats = self.illu(y)
        out_uv, out_uv_feats = self.denoise(uv)

        out = self.FuN(x, out_y_feats, out_uv_feats)
        return out_y, out_uv, out

