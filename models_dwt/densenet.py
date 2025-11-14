# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe's modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe's modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import re
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from models_dwt import *
from models_dwt.EPCAM import EPCAM
from models_dwt.Mix_HL_DFE import Mix_HL_DFE

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']



# HWCEP+MixHL_DEF 4
class _DenseLayer(nn.Module):  
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, use_hwcep=False):
        super(_DenseLayer, self).__init__()
        self.use_hwcep = use_hwcep
        self.drop_rate = drop_rate
        
 
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        

        if use_hwcep:
            self.hwcep = EPCAM(input_dimension=growth_rate)
            

            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(growth_rate, growth_rate, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
            

        if self.use_hwcep:
            hwcep_out = self.hwcep(out)
            gate = self.fusion_gate(out)
            out = gate * hwcep_out + (1 - gate) * out  
        
        # 密集连接
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, use_hwcep=False):
        super(_DenseBlock, self).__init__()
        half_layers = num_layers // 2
        
        for i in range(num_layers):
            use_hwcep_for_this_layer = False
            if i >= half_layers:  
                if (i - half_layers) % 2 == 0:
                    use_hwcep_for_this_layer = use_hwcep
            
            layer = _DenseLayer(
                num_input_features=(num_input_features + i * growth_rate), 
                growth_rate=growth_rate, 
                bn_size=bn_size, 
                drop_rate=drop_rate,
                use_hwcep=use_hwcep_for_this_layer
            )
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, wavename='haar', use_mix_hl_dfe=False):
        super(_Transition, self).__init__()
        self.use_mix_hl_dfe = use_mix_hl_dfe
        
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = Downsample(wavename=wavename)

        if use_mix_hl_dfe:
            self.mix_hl_dfe = Mix_HL_DFE(high_alpha=0.05, low_alpha=0.95)
            
            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_output_features, num_output_features, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        # 基本处理
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        
        # 保存标准转换结果（实现残差连接）
        standard_out = self.pool(out)
        
        # 条件频域增强
        if self.use_mix_hl_dfe:
            # 频域特征处理
            freq_out = self.mix_hl_dfe(out)
            freq_out = self.pool(freq_out)  # 确保维度匹配
            
            # 自适应融合
            gate = self.fusion_gate(standard_out)
            return gate * freq_out + (1 - gate) * standard_out
        else:
            return standard_out


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=100,
                 wavename='haar', pool_only=True, use_hwcep=True, use_mix_hl_dfe=True):

        super(DenseNet, self).__init__()

        # First convolution
        if(pool_only):
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                # 保留初始HWCEP - 对输入特征最初处理非常重要
                ('hwcep0', EPCAM(input_dimension=num_init_features)),
                ('pool0', Downsample(wavename=wavename)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('hwcep0', EPCAM(input_dimension=num_init_features)),
                ('ds0', Downsample(wavename=wavename)),
                ('pool0', Downsample(wavename=wavename)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            use_mix_hl_here = use_mix_hl_dfe and i >= 2
            if use_mix_hl_here:
                self.features.add_module(f'mix_hl_dfe{i}', Mix_HL_DFE(high_alpha=0.05, low_alpha=0.95))
                
            use_hwcep_here = use_hwcep and (i > 0)  
                
            block = _DenseBlock(
                num_layers=num_layers, 
                num_input_features=num_features,
                bn_size=bn_size, 
                growth_rate=growth_rate, 
                drop_rate=drop_rate,
                use_hwcep=use_hwcep_here
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                use_mix_hl_in_transition = use_mix_hl_dfe and i >= 1 
                trans = _Transition(
                    num_input_features=num_features, 
                    num_output_features=num_features // 2, 
                    wavename=wavename,
                    use_mix_hl_dfe=use_mix_hl_in_transition
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        

        if use_hwcep:
            self.features.add_module('hwcep_final', EPCAM(input_dimension=num_features))
            
        if use_mix_hl_dfe:
            self.features.add_module('mix_hl_dfe_final', Mix_HL_DFE(high_alpha=0.05, low_alpha=0.95))
        
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    nn.init.kaiming_normal_(m.weight)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        
        out = self.multi_scale_fusion(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = model_zoo.load_url(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def densenet121(wavename='haar', pool_only=True, use_hwcep=True, use_mix_hl_dfe=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                    wavename=wavename, pool_only=pool_only, 
                    use_hwcep=use_hwcep, use_mix_hl_dfe=use_mix_hl_dfe, **kwargs)
    return model


def densenet169(wavename='haar', pool_only=True, use_hwcep=True, use_mix_hl_dfe=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                    wavename=wavename, pool_only=pool_only, 
                    use_hwcep=use_hwcep, use_mix_hl_dfe=use_mix_hl_dfe, **kwargs)
    return model


def densenet201(wavename='haar', pool_only=True, use_hwcep=True, use_mix_hl_dfe=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                    wavename=wavename, pool_only=pool_only, 
                    use_hwcep=use_hwcep, use_mix_hl_dfe=use_mix_hl_dfe, **kwargs)
    return model


def densenet161(wavename='haar', pool_only=True, use_hwcep=True, use_mix_hl_dfe=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                    wavename=wavename, pool_only=pool_only, 
                    use_hwcep=use_hwcep, use_mix_hl_dfe=use_mix_hl_dfe, **kwargs)
    return model