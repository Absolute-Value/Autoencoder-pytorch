import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_c=3, hidden_dim=128):
        super(AutoEncoder, self).__init__()

        dim = hidden_dim // 8

        self.enc11 = EncodeBlock(1 * in_c, 1 * dim, use_bn=True)
        self.enc12 = EncodeBlock(1 * dim, 1 * dim, use_bn=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.enc21 = EncodeBlock(1 * dim, 2 * dim, use_bn=True)
        self.enc22 = EncodeBlock(2 * dim, 2 * dim, use_bn=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.enc31 = EncodeBlock(2 * dim, 2 * dim, use_bn=True)
        self.enc32 = EncodeBlock(2 * dim, 4 * dim, use_bn=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.enc41 = EncodeBlock(4 * dim, 4 * dim, use_bn=True)
        self.enc42 = EncodeBlock(4 * dim, 8 * dim, use_bn=True)
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        self.enc51 = EncodeBlock(8 * dim, 8 * dim, use_bn=True)

        self.dec51 = DecodeBlock(8 * dim, 8 * dim, use_bn=True)
        self.unpool4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec42 = DecodeBlock(8 * dim, 8 * dim, use_bn=True)
        self.dec41 = DecodeBlock(8 * dim, 4 * dim, use_bn=True)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec32 = DecodeBlock(4 * dim, 4 * dim, use_bn=True)
        self.dec31 = DecodeBlock(4 * dim, 2 * dim, use_bn=True)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec22 = DecodeBlock(2 * dim, 2 * dim, use_bn=True)
        self.dec21 = DecodeBlock(2 * dim, 1 * dim, use_bn=True)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec12 = DecodeBlock(1 * dim, 1 * dim, use_bn=True)
        self.dec11 = nn.ConvTranspose2d(1 * dim, in_c, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # Encode
        enc1 = self.enc12(self.enc11(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc22(self.enc21(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc32(self.enc31(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc42(self.enc41(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc51(pool4)

        # Decode
        dec5 = self.dec51(enc5)

        unpool4 = self.unpool4(dec5)
        dec4 = self.dec41(self.dec42(unpool4))

        unpool3 = self.unpool4(dec4)
        dec3 = self.dec31(self.dec32(unpool3))

        unpool2 = self.unpool2(dec3)
        dec2 = self.dec21(self.dec22(unpool2))

        unpool1 = self.unpool1(dec2)
        dec1 = self.dec11(self.dec12(unpool1))

        x = dec1

        return x

class EncodeBlock(nn.Module):
    def __init__(self, in_feature, out_future, use_bn):
        super(EncodeBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_future

        layers = []
        layers.append(nn.Conv2d(in_feature, out_future, kernel_size=3, stride=1, padding=1, bias=use_bn))
        if use_bn: layers.append(nn.BatchNorm2d(out_future))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_feature, out_future, use_bn):
        super(DecodeBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_future

        layers = []
        layers.append(nn.ConvTranspose2d(in_feature, out_future, kernel_size=3, stride=1, padding=1, bias=use_bn))
        if use_bn: layers.append(nn.BatchNorm2d(out_future))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
