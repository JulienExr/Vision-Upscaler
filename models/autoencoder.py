import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 

def gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return torch.nn.GroupNorm(g, c)
    return torch.nn.GroupNorm(1, c)

class Autoencoder(nn.Module):
    def __init__(self,encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        encoded, features_enc = self.encoder(x)
        return encoded, features_enc
    
    def decode(self, z, features_enc, original_input):
        decoded = self.decoder(z, features_enc, original_input)
        return decoded
    
    def forward(self, x):
        encoded, features_enc = self.encode(x)
        decoded = self.decode(encoded, features_enc, x)
        return decoded

class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = gn(channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = gn(channels)
        self.act2 = nn.SiLU()
    
    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x + residual)

        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.gn = gn(out_channels)
        self.act = nn.SiLU()
        self.res = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.res(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn = gn(out_channels)
        self.act = nn.SiLU()
        self.res = ResBlock(out_channels)

    def forward(self, x, skip_connection):
        
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)

        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.res(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):                  # in x 256 x 256 
        super(Encoder, self).__init__()
        self.down_block1 = DownBlock(in_channels, 32) # 32 x 128 x 128

        self.down_block2 = DownBlock(32, 64)          # 64 x 64 x 64

        self.trans_bottleneck = nn.Sequential(        # 128 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            gn(128),
            nn.SiLU()
        )

        self.resblocks_bottleneck = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )


    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)

        x_bot = self.trans_bottleneck(x2)
        x_bot = self.resblocks_bottleneck(x_bot)
        return x_bot, (x1, x2)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):              # 128 x 64 x 64
        super(Decoder, self).__init__()

        self.up_block1 = UpBlock(128, 32, 64)                   # 64 x 128 x 128

        self.up_block2 = UpBlock(64, in_channels, 32)           # 32 x 256 x 256 

        self.final_conv = nn.Conv2d(32, out_channels, stride=1, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()
    
    def forward(self, x, features_enc, original_input):
        (x1, x2) = features_enc

        x = self.up_block1(x, x1)

        x = self.up_block2(x, original_input)

        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x
    


if __name__ == "__main__":
    channels = 3
    img_size = 256
    
    enc = Encoder(channels)
    dec = Decoder(channels, channels)
    model = Autoencoder(enc, dec)
    
    dummy_input = torch.randn(1, channels, img_size, img_size)
    
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("\nSuccess: dimensions match")
    else:
        print("\nError: Dimensions do not match")