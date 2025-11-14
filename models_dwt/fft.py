import torch
from torch import fft


def fft_feature_ranking(x):

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # 计算傅里叶变换后的特征对任务的贡献（假设任务是分类）
    task_contributions = torch.abs(x_freq.mean(dim=0))  # 取平均值来估计贡献度

    # 按贡献度排序特征
    sorted_indices = torch.argsort(task_contributions, descending=True)

    # 保留前N个特征（频率）
    top_n_freq_indices = sorted_indices[:64]

    # 根据保留的频率索引重建特征
    features_reconstructed = torch.zeros_like(x_freq)
    features_reconstructed[:, top_n_freq_indices, :, :] = x_freq[:, top_n_freq_indices, :, :]

    # 将重建的特征进行逆傅里叶变换
    features_reconstructed = fft.ifftshift(features_reconstructed, dim=(-2, -1))
    features_reconstructed = fft.ifftn(features_reconstructed, dim=(-2, -1)).real

    return features_reconstructed

if __name__ == '__main__':
    x = torch.randn(4, 64, 56, 56)
    model = fft_feature_ranking(x)
    print(model.shape)