import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#SimAM
class SimAM(torch.nn.Module):
    def __init__(self, c_num: int, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

from torch import fft
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered



class ince_conv(nn.Module):
    def __init__(self,model1,model2):
        super(ince_conv, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.downsample1 = nn.Sequential(
        #     nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
        #                                  )
        # self.downsample2 = nn.Sequential(
        #     nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
        #                                  )
        self.GAP = nn.AvgPool2d(kernel_size=2,stride=2)
        self.GMP = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att1 = SimAM(c_num = 768)
        self.att2 = SimAM(c_num = 2048*2+768)
        self.gap1 = nn.AvgPool2d(kernel_size=28)
        self.gap2 = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten(1)
        self.head1 = nn.Linear(768, 3)
        self.head2 = nn.Linear(2048*2+768, 3)
        # self.head3 = nn.Linear(2048 * 2, 3)

    def forward(self, x):
        x1 = self.model1.forward_features0(x)
        x3 = x1
        x2 = self.model2.forward0(x)
        x4 = x2
        x5 = torch.cat([x3,x4],dim=1)
        x5 = Fourier_filter(x5, threshold=1, scale=0.6)
        x5 = self.att1(x5)
        x5 = self.gap1(x5)
        x5_1 = self.flatten(x5)
        x5_2 = self.head1(x5_1)

        x1 = self.model1.forward_features(x1)
        x2 = self.model2.forward(x2)
        x6 = torch.cat([x1,x2],dim=1)
        x6 = self.att2(x6)
        x6 = self.gap2(x6)
        x6 = self.flatten(torch.cat([x6,x5],dim=1))
        x6 = self.head2(x6)

        return x6+x5_2