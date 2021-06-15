import torch
import torch.nn as nn
import torch.nn.functional as F

class segnet(nn.Module):
    def __init__(self, out_channel=10):
        super(segnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.2)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)



class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)
        

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([
            center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

class ResidualBlock(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super().__init__()
        self.recnn = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.recnn(x)
        return x+x1

class ReUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=10,t=2):
        super().__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.rdec1 = ResidualBlock(ch_in=img_ch,ch_out=64,t=t)

        self.rdec2 = ResidualBlock(ch_in=64,ch_out=128,t=t)
        
        self.rdec3 = ResidualBlock(ch_in=128,ch_out=256,t=t)
        
        self.rdec4 = ResidualBlock(ch_in=256,ch_out=512,t=t)
        
        self.rdec5 = ResidualBlock(ch_in=512,ch_out=1024,t=t)
        

        self.upconv5 = up_conv(ch_in=1024,ch_out=512)
        self.renc5 = ResidualBlock(ch_in=1024, ch_out=512,t=t)
        
        self.upconv4 = up_conv(ch_in=512,ch_out=256)
        self.renc4 = ResidualBlock(ch_in=512, ch_out=256,t=t)
        
        self.upconv3 = up_conv(ch_in=256,ch_out=128)
        self.renc3 = ResidualBlock(ch_in=256, ch_out=128,t=t)
        
        self.upconv2 = up_conv(ch_in=128,ch_out=64)
        self.renc2 = ResidualBlock(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.rdec1(x)

        x2 = self.Maxpool(x1)
        x2 = self.rdec2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.rdec3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.rdec4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.rdec5(x5)

        # decoding + concat path
        d5 = self.upconv5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.renc5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.renc4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.renc3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.renc2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())