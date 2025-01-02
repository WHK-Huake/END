import torch
import torch.nn as nn
import torch.nn.functional as TF
from models.common import Conv, CAB


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


class PEN(nn.Module):
    def __init__(self, channel, n_feats):
        super(PEN, self).__init__()
        self.enc1 = nn.Sequential(Conv_Relu(channel, n_feats, 3, 1), Conv(n_feats, n_feats, 3))
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(Conv_Relu(n_feats, n_feats * 2, 3, 1), Conv(n_feats * 2, n_feats * 2, 3))
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv_Relu(n_feats * 2, n_feats * 4, 3, 1)
        self.conv3 = Conv(n_feats * 4, n_feats * 4, 3)

        self.up32 = DeConv_Relu(n_feats * 4, n_feats * 2, 3, 1)
        self.dec2 = nn.Sequential(Conv_Relu(n_feats * 4, n_feats * 4, 3, 1), Conv(n_feats * 4, n_feats * 2, 3))

        self.up21 = DeConv_Relu(n_feats * 2, n_feats, 3, 1)
        self.dec1 = nn.Sequential(Conv_Relu(n_feats * 2, n_feats * 2, 3, 1), Conv(n_feats * 2, n_feats, 3))

        self.out2 = PCM(channel, n_feats, 3, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pooling1(enc1))
        neck = self.enc3(self.pooling2(enc2))
        neck = self.conv3(neck)

        up2 = self.up32(neck)
        cat2 = torch.cat([up2, enc2], 1)
        dec2 = self.dec2(cat2)

        up1 = self.up21(dec2)
        cat1 = torch.cat([up1, enc1], 1)
        dec1 = self.dec1(cat1)

        feats, out = self.out2(dec1,x)
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

        self.illu = PEN(channel=1, n_feats=64)      # L-PEN
        self.denoise = PEN(channel=2, n_feats=64)   # C-PEN
        self.FuN = IEN(3, 64, 64)

    def forward(self, x):
        y = x[:, 0, :, :].unsqueeze(1)
        uv = x[:, 1:3, :, :]

        out_y, out_y_feats = self.illu(y)
        out_uv, out_uv_feats = self.denoise(uv)

        out = self.FuN(x, out_y_feats, out_uv_feats)
        return out_y, out_uv, out

