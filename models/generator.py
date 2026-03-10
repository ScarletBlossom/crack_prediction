import torch
import torch.nn as nn

from .blocks import BigConv, DownPool, DownSample, OutputBlock, UpSample, UpSampleBilinear, init_weights


class Generator(nn.Module):
    """Full-scale skip generator, behavior preserved from the original code."""

    def __init__(self):
        super().__init__()
        self.xe2 = DownSample(3, 32, 4, 2)
        self.xe3 = DownSample(32, 64, 4, 2)
        self.xe4 = DownSample(64, 128, 4, 2)
        self.xe5 = DownSample(128, 128, 4, 2)
        self.xe6 = DownSample(128, 256, 4, 2)
        self.xe7 = DownSample(256, 256, 4, 2)
        self.xe8 = DownSample(256, 512, 4, 2)
        self.xe9 = DownSample(512, 512, 4, 2)

        self.x81 = DownPool(3, 32, 128)
        self.x82 = DownPool(32, 32, 64)
        self.x83 = DownPool(64, 32, 32)
        self.x84 = DownPool(128, 32, 16)
        self.x85 = DownPool(128, 32, 8)
        self.x86 = DownPool(256, 32, 4)
        self.x87 = DownPool(256, 32, 2)
        self.x88 = DownSample(512, 32, 3, 1)
        self.x89 = UpSample(512, 32, 4, 2, dropout=True)
        self.xd8 = BigConv(288, 288)

        self.x71 = DownPool(3, 32, 64)
        self.x72 = DownPool(32, 32, 32)
        self.x73 = DownPool(64, 32, 16)
        self.x74 = DownPool(128, 32, 8)
        self.x75 = DownPool(128, 32, 4)
        self.x76 = DownPool(256, 32, 2)
        self.x77 = DownSample(256, 32, 3, 1)
        self.x78 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x79 = UpSampleBilinear(512, 32, 4)
        self.xd7 = BigConv(288, 288)

        self.x61 = DownPool(3, 32, 32)
        self.x62 = DownPool(32, 32, 16)
        self.x63 = DownPool(64, 32, 8)
        self.x64 = DownPool(128, 32, 4)
        self.x65 = DownPool(128, 32, 2)
        self.x66 = DownSample(256, 32, 3, 1)
        self.x67 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x68 = UpSampleBilinear(1440, 32, 4)
        self.x69 = UpSampleBilinear(512, 32, 8)
        self.xd6 = BigConv(288, 288)

        self.x51 = DownPool(3, 32, 16)
        self.x52 = DownPool(32, 32, 8)
        self.x53 = DownPool(64, 32, 4)
        self.x54 = DownPool(128, 32, 2)
        self.x55 = DownSample(128, 32, 3, 1)
        self.x56 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x57 = UpSampleBilinear(1440, 32, 4)
        self.x58 = UpSampleBilinear(1440, 32, 8)
        self.x59 = UpSampleBilinear(512, 32, 16)
        self.xd5 = BigConv(288, 288)

        self.x41 = DownPool(3, 32, 8)
        self.x42 = DownPool(32, 32, 4)
        self.x43 = DownPool(64, 32, 2)
        self.x44 = DownSample(128, 32, 3, 1)
        self.x45 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x46 = UpSampleBilinear(1440, 32, 4)
        self.x47 = UpSampleBilinear(1440, 32, 8)
        self.x48 = UpSampleBilinear(1440, 32, 16)
        self.x49 = UpSampleBilinear(512, 32, 32)
        self.xd4 = BigConv(288, 288)

        self.x31 = DownPool(3, 32, 4)
        self.x32 = DownPool(32, 32, 2)
        self.x33 = DownSample(64, 32, 3, 1)
        self.x34 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x35 = UpSampleBilinear(1440, 32, 4)
        self.x36 = UpSampleBilinear(1440, 32, 8)
        self.x37 = UpSampleBilinear(1440, 32, 16)
        self.x38 = UpSampleBilinear(1440, 32, 32)
        self.x39 = UpSampleBilinear(512, 32, 64)
        self.xd3 = BigConv(288, 288)

        self.x21 = DownPool(3, 32, 2)
        self.x22 = DownSample(32, 32, 3, 1)
        self.x23 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x24 = UpSampleBilinear(1440, 32, 4)
        self.x25 = UpSampleBilinear(1440, 32, 8)
        self.x26 = UpSampleBilinear(1440, 32, 16)
        self.x27 = UpSampleBilinear(1440, 32, 32)
        self.x28 = UpSampleBilinear(1440, 32, 64)
        self.x29 = UpSampleBilinear(512, 32, 128)
        self.xd2 = BigConv(288, 288)

        self.x11 = DownSample(3, 32, 3, 1)
        self.x12 = UpSample(1440, 32, 4, 2, dropout=True)
        self.x13 = UpSampleBilinear(1440, 32, 4)
        self.x14 = UpSampleBilinear(1440, 32, 8)
        self.x15 = UpSampleBilinear(1440, 32, 16)
        self.x16 = UpSampleBilinear(1440, 32, 32)
        self.x17 = UpSampleBilinear(1440, 32, 64)
        self.x18 = UpSampleBilinear(1440, 32, 128)
        self.x19 = UpSampleBilinear(512, 32, 256)
        self.xd1 = BigConv(288, 288)

        self.last = OutputBlock(1440, 3)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xe1 = x
        xe2 = self.xe2(xe1)
        xe3 = self.xe3(xe2)
        xe4 = self.xe4(xe3)
        xe5 = self.xe5(xe4)
        xe6 = self.xe6(xe5)
        xe7 = self.xe7(xe6)
        xe8 = self.xe8(xe7)
        xe9 = self.xe9(xe8)

        x81, x82, x83, x84 = self.x81(xe1), self.x82(xe2), self.x83(xe3), self.x84(xe4)
        x85, x86, x87, x88, x89 = self.x85(xe5), self.x86(xe6), self.x87(xe7), self.x88(xe8), self.x89(xe9)
        xd8 = self.xd8(torch.cat([x89, x88, x87, x86, x85, x84, x83, x82, x81], dim=1))

        x71, x72, x73, x74 = self.x71(xe1), self.x72(xe2), self.x73(xe3), self.x74(xe4)
        x75, x76, x77, x78, x79 = self.x75(xe5), self.x76(xe6), self.x77(xe7), self.x78(xd8), self.x79(xe9)
        xd7 = self.xd7(torch.cat([x79, x78, x77, x76, x75, x74, x73, x72, x71], dim=1))

        x61, x62, x63, x64 = self.x61(xe1), self.x62(xe2), self.x63(xe3), self.x64(xe4)
        x65, x66, x67, x68, x69 = self.x65(xe5), self.x66(xe6), self.x67(xd7), self.x68(xd8), self.x69(xe9)
        xd6 = self.xd6(torch.cat([x69, x68, x67, x66, x65, x64, x63, x62, x61], dim=1))

        x51, x52, x53, x54 = self.x51(xe1), self.x52(xe2), self.x53(xe3), self.x54(xe4)
        x55, x56, x57, x58, x59 = self.x55(xe5), self.x56(xd6), self.x57(xd7), self.x58(xd8), self.x59(xe9)
        xd5 = self.xd5(torch.cat([x59, x58, x57, x56, x55, x54, x53, x52, x51], dim=1))

        x41, x42, x43, x44 = self.x41(xe1), self.x42(xe2), self.x43(xe3), self.x44(xe4)
        x45, x46, x47, x48, x49 = self.x45(xd5), self.x46(xd6), self.x47(xd7), self.x48(xd8), self.x49(xe9)
        xd4 = self.xd4(torch.cat([x49, x48, x47, x46, x45, x44, x43, x42, x41], dim=1))

        x31, x32, x33 = self.x31(xe1), self.x32(xe2), self.x33(xe3)
        x34, x35, x36, x37, x38, x39 = self.x34(xd4), self.x35(xd5), self.x36(xd6), self.x37(xd7), self.x38(xd8), self.x39(xe9)
        xd3 = self.xd3(torch.cat([x39, x38, x37, x36, x35, x34, x33, x32, x31], dim=1))

        x21, x22 = self.x21(xe1), self.x22(xe2)
        x23, x24, x25, x26, x27, x28, x29 = self.x23(xd3), self.x24(xd4), self.x25(xd5), self.x26(xd6), self.x27(xd7), self.x28(xd8), self.x29(xe9)
        xd2 = self.xd2(torch.cat([x29, x28, x27, x26, x25, x24, x23, x22, x21], dim=1))

        x11 = self.x11(xe1)
        x12, x13, x14, x15, x16, x17, x18, x19 = self.x12(xd2), self.x13(xd3), self.x14(xd4), self.x15(xd5), self.x16(xd6), self.x17(xd7), self.x18(xd8), self.x19(xe9)
        xd1 = self.xd1(torch.cat([x19, x18, x17, x16, x15, x14, x13, x12, x11], dim=1))
        return self.last(xd1)
