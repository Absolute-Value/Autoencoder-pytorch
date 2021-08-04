import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self, compression_rate = 1, filter_num = 1):
        super(CAE, self).__init__()

        self.compression_rate = compression_rate
        self.filter = filter_num

        # https://arxiv.org/pdf/1812.02765.pdf.
        # モデルの作成。
        self.encoder = nn.Sequential(
            Block(3, 16),
            nn.MaxPool2d(2, 2),
            Block(16, 32),
            nn.MaxPool2d(2, 2),
            Block(32, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            nn.MaxPool2d(2, 2),
            Block(128, 256),
            nn.MaxPool2d(2, 2),
            Block(256, 512),
            nn.MaxPool2d(2, 2),
            Block(512, 512),

            nn.Conv2d(512, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*64, 2048), 
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2048, 8*8*64), 
            nn.ReLU(),
            Reshape(-1, 64, 8, 8),
            nn.Conv2d(64, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            Block(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            Block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            Block(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            Block(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            Block(16, 16),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2), 
            #Block(3, 3),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
        """
        self.encoder = nn.Sequential(
            Reshape(-1,28*28),
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            Reshape(-1, 1, 28, 28))
        """

    def forward(self, x):
        z = self.encoder(x)
        x_pred = self.decoder(z)

        return x_pred
        #return z

# make tensor.view() Module to use it in Sequential
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4
        # 1x1 の畳み込み
        self.conv1 = nn.Conv2d(channel_in, channel,
                               kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()
        # 3x3 の畳み込み
        self.conv2 = nn.Conv2d(channel, channel,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()
        # 1x1 の畳み込み
        self.conv3 = nn.Conv2d(channel, channel_out,
                               kernel_size=(1, 1),
                               padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)
        # skip connection用のチャネル数調整        
        self.shortcut = self._shortcut(channel_in, channel_out)
        
        self.relu3 = nn.ReLU()
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        return y
    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out,
                         kernel_size=(1, 1),
                         padding=0)
