import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Network [{type(self).__name__}] was created. Total number of parameters: {num_params / 1e6:.1f} million.")

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight"):
                if "Conv" in classname or "Linear" in classname:
                    if init_type == "normal":
                        nn.init.normal_(m.weight, 0.0, gain)
                    elif init_type == "xavier":
                        nn.init.xavier_normal_(m.weight, gain=gain)
                    elif init_type == "kaiming":
                        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                    elif init_type == "orthogonal":
                        nn.init.orthogonal_(m.weight, gain=gain)
                    else:
                        raise NotImplementedError(f"Initialization method [{init_type}] is not implemented.")
                elif "InstanceNorm2d" in classname:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        self.apply(init_func)

    def forward(self, x):
        raise NotImplementedError("forward method must be implemented in subclasses")


class InpaintGenerator(BaseNetwork):
    def __init__(self, args):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.add_module(
                f"block{i:02d}",
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [getattr(self, f"block{i:02d}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class Discriminator(BaseNetwork):
    def __init__(self):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

    def forward(self, x):
        return self.conv(x)