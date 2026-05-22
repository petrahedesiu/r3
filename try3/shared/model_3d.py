
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock3d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out


class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_filters: int = 16,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = base_filters
        channels = [f, f * 2, f*4, f * 8]

        self.enc1 = ResConvBlock3d(in_channels, channels[0])
        self.enc2 = ResConvBlock3d(channels[0], channels[1])
        self.enc3 = ResConvBlock3d(channels[1], channels[2])
        self.enc4 = ResConvBlock3d(channels[2], channels[3])

        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ResConvBlock3d(channels[3], channels[3])

        self.dec4 = ResConvBlock3d(channels[3] + channels[3], channels[3])
        self.dec3 = ResConvBlock3d(channels[3] + channels[2], channels[2])
        self.dec2 = ResConvBlock3d(channels[2] + channels[1], channels[1])
        self.dec1 = ResConvBlock3d(channels[1] + channels[0], channels[0])

        self.out_conv = nn.Conv3d(channels[0], num_classes, 1)

        if self.deep_supervision:
            self.ds_out2 = nn.Conv3d(channels[1], num_classes, 1)
            self.ds_out3 = nn.Conv3d(channels[2], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _upsample_like(self, x, target):
        return F.interpolate(
            x, size=target.shape[2:], mode="trilinear", align_corners=False
        )

    def forward(self, x):
        # encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._upsample_like(bn, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._upsample_like(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._upsample_like(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._upsample_like(d2, e1), e1], dim=1))

        main_out = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds2 = self.ds_out2(d2)
            ds2 = self._upsample_like(ds2, main_out)
            ds3 = self.ds_out3(d3)
            ds3 = self._upsample_like(ds3, main_out)
            return main_out, ds2, ds3

        return main_out


def create_3d_model(
    in_channels: int = 3,
    num_classes: int = 3,
    base_filters: int = 32,
    deep_supervision: bool = True,
) -> UNet3D:
    return UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        deep_supervision=deep_supervision,
    )
