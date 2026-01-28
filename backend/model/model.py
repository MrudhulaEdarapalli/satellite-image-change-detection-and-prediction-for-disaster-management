import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SiameseUNet, self).__init__()

        # Shared Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck (applied on the concatenated difference or just difference)
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512) # 512 + 512 (skip)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256) # 256 + 256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128) # 128 + 128

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64) # 64 + 64

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_step(self, x):
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        p3 = self.pool(s3)
        s4 = self.enc4(p3)
        p4 = self.pool(s4)
        return s1, s2, s3, s4, p4

    def forward(self, x1, x2):
        # Step 1 & 2: Shared Encoder for both images
        s1_1, s1_2, s1_3, s1_4, p1_4 = self.encoder_step(x1)
        s2_1, s2_2, s2_3, s2_4, p2_4 = self.encoder_step(x2)

        # Step 3: Feature Comparison (Absolute Difference)
        diff1 = torch.abs(s1_1 - s2_1)
        diff2 = torch.abs(s1_2 - s2_2)
        diff3 = torch.abs(s1_3 - s2_3)
        diff4 = torch.abs(s1_4 - s2_4)
        diff_p = torch.abs(p1_4 - p2_4)

        # Step 4: Decoder (U-Net Part)
        b = self.bottleneck(diff_p)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat((u4, diff4), dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat((u3, diff3), dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat((u2, diff2), dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat((u1, diff1), dim=1))

        # Step 5: Output Mask
        output = self.final_conv(d1)
        return output

if __name__ == "__main__":
    model = SiameseUNet()
    x1 = torch.randn(1, 3, 96, 96)
    x2 = torch.randn(1, 3, 96, 96)
    y = model(x1, x2)
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {y.shape}")
