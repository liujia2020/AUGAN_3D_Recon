import torch
import torch.nn as nn
import functools

class UnetGenerator3D(nn.Module):
    """
    3D U-Net 生成器 (支持动态深度与递归构建)。
    
    设计理念：
    - 既然是 3D 各向异性数据，我们允许通过 'num_downs' 参数灵活控制下采样次数。
    - 采用 'UnetSkipConnectionBlock3D' 递归构建，避免硬编码每一层。
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        """
        参数:
            input_nc (int):  输入通道数 (通常为1)
            output_nc (int): 输出通道数 (通常为1)
            num_downs (int): 下采样次数。例如 6 代表缩小 2^6 = 64 倍。
                             对于 Patch (256, 64, 64)，num_downs=6 是极限 (64->1)。
            ngf (int):       第一层卷积的滤波器数量 (默认 64)
            norm_layer:      归一化层 (默认 InstanceNorm3d)
            use_dropout:     是否在中间层使用 Dropout
        """
        super(UnetGenerator3D, self).__init__()
        
        # 构造 U-Net 是从“最里面” (Bottleneck) 开始，一层层往外包的。
        
        # 1. 最里层 (Innermost)
        # 它的输入和输出都是 ngf * 8，它是 U-Net 的最底部，不进行下采样了，而是直接反卷积回来。
        unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=None, 
                                               norm_layer=norm_layer, innermost=True)
        
        # 2. 中间层 (Intermediate)
        # 我们需要循环构建 (num_downs - 5) 次。
        # 比如 num_downs=6，这里循环 1 次。
        # 这些层通道数保持最大 (ngf * 8)
        for i in range(num_downs - 5): 
            unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                                                   norm_layer=norm_layer, use_dropout=use_dropout)
        
        # 3. 逐步上采样层 (Up-sampling)
        # 这里的通道数开始减少: 512 -> 256 -> 128 -> 64
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # 4. 最外层 (Outermost)
        # 输入是图像 (input_nc)，输出是图像 (output_nc)
        self.model = UnetSkipConnectionBlock3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, 
                                               outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock3D(nn.Module):
    """
    定义包含跳跃连接 (Skip Connection) 的 3D U-Net 子模块。
    结构: [Down] -> [Submodule] -> [Up]
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock3D, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        
        # 确定是否使用 Bias (如果是 InstanceNorm，不需要 Bias，因为 Norm 层有仿射参数)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        if input_nc is None:
            input_nc = outer_nc

        # --- 3D 卷积组件 ---
        # 降采样: Stride=2
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        
        # 激活函数与归一化
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # --- 构建逻辑 (递归核心) ---
        
        if outermost:
            # 最外层: 输入图像 -> [Submodule] -> 输出图像
            # 最后一层使用 Tanh 激活，将像素值映射回 [-1, 1]
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            
        elif innermost:
            # 最里层: 只有降采样和上采样，没有 Submodule
            # Skip Connection 这里是直接拼接
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            
        else:
            # 中间层: [Down] -> [Submodule] -> [Up]
            # 这里的输入会包含上一层的 Skip Connection，所以是 inner_nc * 2
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # 标准 U-Net 跳跃连接: 将输入 x 与处理后的结果拼接 (Concat)
            # x 的形状是 [Batch, C, D, H, W]
            # model(x) 的形状也是 [Batch, C_out, D, H, W] (因为是对称结构)
            # 在 Channel 维度 (dim=1) 拼接
            return torch.cat([x, self.model(x)], 1)