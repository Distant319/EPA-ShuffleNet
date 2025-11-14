import torch
import torch.nn as nn

class EPCAM(nn.Module):
    def __init__(self, input_dimension, use_bias=False, operation_mode='fc'):  
        super().__init__()  

        self.channel_normalization = nn.BatchNorm2d(input_dimension)

        self.horizontal_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  
        self.vertical_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  

        self.channel_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  
        self.horizontal_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (1, 7), stride=1,
                                                  padding=(0, 7 // 2), groups=input_dimension, bias=False) 
        self.vertical_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (7, 1), stride=1,
                                                 padding=(7 // 2, 0), groups=input_dimension, bias=False)  

        self.operation_mode = operation_mode  
        if operation_mode == 'fc':  
            self.horizontal_phase_calculator = nn.Sequential(  
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(inplace=False)  
            )
            self.vertical_phase_calculator = nn.Sequential(  
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(inplace=False)  
            )
        else: 
            self.horizontal_phase_calculator = nn.Sequential(  
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(inplace=False) 
            )
            self.vertical_phase_calculator = nn.Sequential(  
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(inplace=False)  
            )

        self.feature_fusion_layer = nn.Sequential(
            nn.Conv2d(4 * input_dimension, input_dimension, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_dimension),
            nn.ReLU(inplace=False),
        )

    def forward(self, input_tensor):  
        horizontal_phase = self.horizontal_phase_calculator(input_tensor)  
        vertical_phase = self.vertical_phase_calculator(input_tensor)  

        horizontal_amplitude = self.horizontal_feature_conv(input_tensor) 
        vertical_amplitude = self.vertical_feature_conv(input_tensor)  

        horizontal_euler = torch.cat([horizontal_amplitude * torch.cos(horizontal_phase),
                                      horizontal_amplitude * torch.sin(horizontal_phase)], dim=1)  
        vertical_euler = torch.cat([vertical_amplitude * torch.cos(vertical_phase),
                                    vertical_amplitude * torch.sin(vertical_phase)], dim=1)  

        original_input = input_tensor
        transformed_h = self.horizontal_transform_conv(horizontal_euler)  
        transformed_w = self.vertical_transform_conv(vertical_euler)  
        transformed_c = self.channel_feature_conv(input_tensor) 

        merged_features = torch.cat([original_input, transformed_h, transformed_w, transformed_c], dim=1)  


        output_tensor = self.feature_fusion_layer(merged_features)
        return output_tensor  



if __name__ == '__main__': 
    euler_processor = EPCAM(input_dimension=64)

