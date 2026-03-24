# Copyright (c) OpenMMLab. All rights reserved.
# 导入必要的库和模块
import math  # 数学运算库
import torch.nn as nn  # PyTorch神经网络模块
# 从mmcv库导入卷积模块、Swish激活函数、构建激活层、常数初始化、Kaiming初始化
from mmcv.cnn import ConvModule, Swish, build_activation_layer, constant_init, kaiming_init
# 从mmcv.runner导入加载检查点的函数
from mmcv.runner import load_checkpoint
# 从mmcv.utils导入批量归一化类
from mmcv.utils import _BatchNorm

# 从本地项目utils模块导入缓存检查点和获取根日志器的函数
from ...utils import cache_checkpoint, get_root_logger
# 从本地项目builder模块导入BACKBONES注册器
from ..builder import BACKBONES

# x3d 可以进行空间维度和时间维度的压缩 ，进行规模化的调参
# 通道注意力模块
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模块。
    它通过学习每个特征通道的重要性来重新校准通道特征响应。
    """

    def __init__(self, channels, reduction): 
        """
        初始化SE模块。

        Args:
            channels (int): 输入特征图的通道数。
            reduction (float): 用于降维的压缩比，瓶颈层的通道数为 channels // reduction。
        """
        super().__init__()  # 调用父类nn.Module的构造函数
        # 创建一个全局平均池化层，将每个通道的空间和时间维度压缩为1x1x1
        self.avg_pool = nn.AdaptiveAvgPool3d(1) 
        # 计算瓶颈层的通道数，即压缩后的通道数
        self.bottleneck = self._round_width(channels, reduction) 
        # 第一个1x1x1卷积层，用于降维，从原始通道数降到瓶颈层通道数
        self.fc1 = nn.Conv3d(
            channels, self.bottleneck, kernel_size=1, padding=0)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个1x1x1卷积层，用于升维，从瓶颈层通道数恢复到原始通道数
        self.fc2 = nn.Conv3d(
            self.bottleneck, channels, kernel_size=1, padding=0)
        # Sigmoid激活函数，输出0到1之间的权重
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8): 
        """
        根据乘数对宽度（通道数）进行四舍五入，确保结果是divisor的倍数且不小于min_width。

        Args:
            width (int): 原始通道数。
            multiplier (float): 压缩比例。
            min_width (int): 最小通道数。默认为8。
            divisor (int): 输出通道数必须是此值的倍数。默认为8。

        Returns:
            int: 四舍五入并满足条件后的通道数。
        """
        width *= multiplier # 将原始通道数乘以压缩比例得到目标通道数
        min_width = min_width or divisor # 如果min_width为0或None，则使用divisor作为最小值
        # 计算最接近width/divisor的整数，然后乘以divisor，实现向下取整到divisor的倍数
        width_out = max(min_width,
                        int(width + divisor / 2) // divisor * divisor)
        # 如果结果远小于目标值（< 0.9 * width），则向上取整到下一个divisor的倍数
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out) # 返回最终的整数通道数

    def forward(self, x):
        """
        定义前向传播过程。

        Args:
            x (Tensor): 形状为 (N, C, T, H, W) 的输入张量。

        Returns:
            Tensor: 与输入x形状相同，但经过通道注意力加权的输出张量。
        """
        module_input = x # 保存原始输入作为残差连接的基础
        x = self.avg_pool(x) # 对输入进行全局平均池化，得到 (N, C, 1, 1, 1)
        x = self.fc1(x) # 降维
        x = self.relu(x) # 激活
        x = self.fc2(x) # 升维
        x = self.sigmoid(x) # 得到每个通道的注意力权重
        # 将计算出的注意力权重与原始输入相乘，实现通道级别的特征重标定
        return module_input * x


class BlockX3D(nn.Module):
    """
    X3D模型的基本构建块。

    Args:
        inplanes (int): 第一个卷积层的输入通道数。
        planes (int): 中间瓶颈层的通道数。
        outplanes (int): 最终输出的通道数。
        spatial_stride (int): 空间维度上的步长。默认: 1。
        downsample (nn.Module | None): 下采样层。默认: None。
        se_ratio (float | None): Squeeze-and-Excitation单元的压缩比。
            如果设置为None，则表示不使用SE单元。默认: None。
        use_swish (bool): 是否在3x3x3卷积前后使用swish激活函数。默认: True。
        conv_cfg (dict): 卷积层的配置字典。默认: ``dict(type='Conv3d')``。
        norm_cfg (dict): 归一化层的配置字典。默认: ``dict(type='BN3d')``。
        act_cfg (dict): 激活函数层的配置字典。默认: ``dict(type='ReLU')``。
    """

    def __init__(self,
                 inplanes,       # 第一个卷积层的输入通道数
                 planes,         # 中间瓶颈层的通道数
                 outplanes,      # 最终输出的通道数
                 spatial_stride=1, # 空间步长
                 downsample=None,  # 下采样层
                 se_ratio=None,    # SE模块的压缩比例
                 use_swish=True,   # 是否使用Swish激活函数
                 conv_cfg=dict(type='Conv3d'), # 卷积层配置
                 norm_cfg=dict(type='BN3d'),   # 归一化层配置
                 act_cfg=dict(type='ReLU')):  # 激活函数配置
        super().__init__() # 调用父类构造函数

        # 保存初始化参数为实例属性，便于后续使用
        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # 定义Swish激活函数的配置
        self.act_cfg_swish = dict(type='Swish')

        # 第一个 1x1x1 卷积：降维（expansion），减少计算量
        self.conv1 = ConvModule(
            in_channels=inplanes,      # 输入通道数
            out_channels=planes,       # 输出通道数
            kernel_size=1,             # 卷积核大小
            stride=1,                  # 步长
            padding=0,                 # 填充
            bias=False,                # 不使用偏置
            conv_cfg=self.conv_cfg,    # 卷积配置
            norm_cfg=self.norm_cfg,    # 归一化配置
            act_cfg=self.act_cfg)      # 激活函数配置
        
        # 3x3x3 深度可分离卷积 (Depthwise Convolution)
        # groups=planes 表示每个输入通道独立进行卷积，大大减少了参数量
        self.conv2 = ConvModule(
            in_channels=planes,        # 输入通道数
            out_channels=planes,       # 输出通道数等于输入，因为是深度可分离卷积
            kernel_size=3,             # 卷积核大小
            stride=(1, self.spatial_stride, self.spatial_stride), # 时间维度步长为1，空间维度步长为spatial_stride
            padding=1,                 # 填充
            groups=planes,             # 设置为输入通道数，实现深度可分离
            bias=False,                # 不使用偏置
            conv_cfg=self.conv_cfg,    # 卷积配置
            norm_cfg=self.norm_cfg,    # 归一化配置
            act_cfg=None)              # 不在此处应用激活函数

        # Swish激活函数或恒等映射（如果use_swish为False）
        self.swish = Swish() if self.use_swish else nn.Identity()

        # 第三个 1x1x1 卷积：升维回正常通道数
        self.conv3 = ConvModule(
            in_channels=planes,        # 输入通道数
            out_channels=outplanes,    # 输出通道数
            kernel_size=1,             # 卷积核大小
            stride=1,                  # 步长
            padding=0,                 # 填充
            bias=False,                # 不使用偏置
            conv_cfg=self.conv_cfg,    # 卷积配置
            norm_cfg=self.norm_cfg,    # 归一化配置
            act_cfg=None)              # 不在此处应用激活函数

        # 如果指定了SE比率，则添加SE模块
        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        # 构建最终的激活函数层
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """定义前向传播的计算过程"""

        def _inner_forward(x):
            """内部前向传播函数，用于利用检查点功能节省内存"""
            identity = x # 保存原始输入作为残差连接

            # 依次通过三个卷积层
            out = self.conv1(x)     # 1x1降维
            out = self.conv2(out)   # 3x3深度可分离卷积
            # 如果使用了SE模块，则进行通道注意力处理
            if self.se_ratio is not None:
                out = self.se_module(out)

            # 应用Swish激活函数
            out = self.swish(out)

            out = self.conv3(out)   # 1x1升维

            # 如果存在下采样层，对原始输入也进行下采样，以匹配输出尺寸
            if self.downsample is not None:
                identity = self.downsample(x)

            # 残差连接：输出 = 卷积分支输出 + 输入分支（可能经过下采样）
            out = out + identity
            return out

        # 执行内部前向传播
        out = _inner_forward(x)
        # 在残差连接后应用最终的激活函数（通常是ReLU）
        out = self.relu(out)
        return out


# 我们不支持使用2D预训练权重初始化X3D
@BACKBONES.register_module() # 将X3D类注册到BACKBONES模块中，方便通过配置文件调用
class X3D(nn.Module):
    """
    X3D骨干网络。论文链接: https://arxiv.org/pdf/2004.04730.pdf.

    Args:
        gamma_w (float): 全局通道宽度扩展因子。默认: 1。
        gamma_b (float): 瓶颈层通道宽度扩展因子。默认: 1。
        gamma_d (float): 网络深度扩展因子。默认: 1。
        pretrained (str | None): 预训练模型的路径名。默认: None。
        in_channels (int): 输入特征的通道数。默认: 3。
        num_stages (int): Resnet阶段数。默认: 4。
        spatial_strides (Sequence[int]): 每个阶段残差块的空间步长。
            默认: ``(1, 2, 2, 2)``。
        frozen_stages (int): 需要冻结的阶段数（所有参数固定）。如果设置为-1，
            则表示不冻结任何参数。默认: -1。
        se_style (str): 将SE模块插入BlockX3D的方式，'half'表示插入一半的块，
            'all'表示插入所有块。默认: 'half'。
        se_ratio (float | None): Squeeze-and-Excitation单元的压缩比。
            如果设置为None，则表示不使用SE单元。默认: 1 / 16。
        use_swish (bool): 是否在3x3x3卷积前后使用swish激活函数。默认: True。
        conv_cfg (dict): 卷积层的配置。必需的键是``type``。
            默认: ``dict(type='Conv3d')``。
        norm_cfg (dict): 归一化层的配置。必需的键是``type``和``requires_grad``。
            默认: ``dict(type='BN3d', requires_grad=True)``。
        act_cfg (dict): 激活函数层的配置字典。
            默认: ``dict(type='ReLU', inplace=True)``。
        norm_eval (bool): 是否将BN层设置为评估模式，即冻结运行统计量（均值和方差）。
            默认: False。
        zero_init_residual (bool): 是否对残差块使用零初始化。
            默认: True。
        kwargs (dict, optional): "make_res_layer"的关键参数。
    """

    def __init__(self,
                 gamma_w=1.0,          # 通道宽度乘子
                 gamma_b=2.25,         # 瓶颈层宽度乘子
                 gamma_d=2.2,          # 深度（层数）乘子
                 pretrained=None,      # 预训练模型路径
                 in_channels=3,        # 输入图像通道数
                 base_channels=24,     # 初始基础通道数
                 num_stages=4,         # 网络阶段数
                 stage_blocks=(1, 2, 5, 3), # 每个阶段的块数量
                 spatial_strides=(2, 2, 2, 2), # 每个阶段第一个块的空间步长
                 frozen_stages=-1,     # 冻结的阶段数
                 se_style='half',      # SE模块插入风格
                 se_ratio=1 / 16,      # SE模块压缩比
                 use_swish=True,       # 是否使用Swish激活
                 conv_cfg=dict(type='Conv3d'), # 卷积配置
                 norm_cfg=dict(type='BN3d', requires_grad=True), # 归一化配置
                 act_cfg=dict(type='ReLU', inplace=True), # 激活函数配置
                 norm_eval=False,      # BN层是否设为eval模式
                 zero_init_residual=True, # 是否零初始化残差
                 **kwargs):            # 其他参数
        super().__init__() # 调用父类构造函数

        # 保存超参数
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d

        self.pretrained = pretrained
        self.in_channels = in_channels
        # 硬编码的基础通道数，可通过gamma_w调整
        self.base_channels = base_channels
        self.stage_blocks = stage_blocks

        # 根据gamma_w调整基础通道数
        self.base_channels = self._round_width(self.base_channels,
                                               self.gamma_w)

        # 根据gamma_d调整每个阶段的块数量
        self.stage_blocks = [
            self._round_repeats(x, self.gamma_d) for x in self.stage_blocks
        ]

        self.num_stages = num_stages
        assert 1 <= num_stages <= 4 # 确保阶段数在合理范围内
        self.spatial_strides = spatial_strides
        assert len(spatial_strides) == num_stages # 确保步长列表长度与阶段数一致
        self.frozen_stages = frozen_stages

        self.se_style = se_style
        assert self.se_style in ['all', 'half'] # 确保SE风格有效
        self.se_ratio = se_ratio
        assert (self.se_ratio is None) or (self.se_ratio > 0) # 确保SE比率有效
        self.use_swish = use_swish

        # 保存配置
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        # 使用BlockX3D作为基本构建块
        self.block = BlockX3D
        # 根据num_stages截断stage_blocks列表
        self.stage_blocks = self.stage_blocks[:num_stages]
        # 记录上一层的输出通道数，用于构建残差连接
        self.layer_inplanes = self.base_channels
        # 构建网络的“茎”部分（stem）
        self._make_stem_layer()

        # 初始化存储各阶段层名称的列表
        self.res_layers = []
        # 循环构建每个阶段
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i] # 获取当前阶段的空间步长
            # 计算当前阶段的输入通道数 (base_channels * 2^i)
            inplanes = self.base_channels * 2**i
            # 计算瓶颈层的中间通道数
            planes = int(inplanes * self.gamma_b)

            # 构建当前阶段的残差层
            res_layer = self.make_res_layer(
                self.block,            # 使用的块类型
                self.layer_inplanes,   # 上一层的输出通道数（用于下采样和残差连接）
                inplanes,              # 当前阶段的输入通道数
                planes,                # 当前阶段瓶颈层的通道数
                num_blocks,            # 当前阶段的块数量
                spatial_stride=spatial_stride, # 空间步长
                se_style=self.se_style,        # SE风格
                se_ratio=self.se_ratio,        # SE比率
                use_swish=self.use_swish,      # 是否使用Swish
                norm_cfg=self.norm_cfg,        # 归一化配置
                conv_cfg=self.conv_cfg,        # 卷积配置
                act_cfg=self.act_cfg,          # 激活函数配置
                **kwargs)                      # 其他参数
            # 更新下一层的输入通道数
            self.layer_inplanes = inplanes
            # 为当前阶段创建一个命名（如'layer1', 'layer2'等）
            layer_name = f'layer{i + 1}'
            # 将构建好的层添加到模型中
            self.add_module(layer_name, res_layer)
            # 将层名添加到列表中，便于后续forward遍历
            self.res_layers.append(layer_name)

        # 计算最终特征图的通道数
        self.feat_dim = self.base_channels * 2**(len(self.stage_blocks) - 1)
        # 添加最后一个1x1x1卷积层，进一步扩展通道数
        self.conv5 = ConvModule(
            self.feat_dim,                   # 输入通道数
            int(self.feat_dim * self.gamma_b), # 输出通道数
            kernel_size=1,                   # 卷积核大小
            stride=1,                        # 步长
            padding=0,                       # 填充
            bias=False,                      # 不使用偏置
            conv_cfg=self.conv_cfg,          # 卷积配置
            norm_cfg=self.norm_cfg,          # 归一化配置
            act_cfg=self.act_cfg)            # 激活函数配置
        # 更新最终特征维度
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        """
        根据宽度乘数对通道数进行四舍五入。

        Args:
            width (int): 原始通道数。
            multiplier (float): 宽度乘数 (gamma_w)。
            min_depth (int): 最小通道数。默认: 8。
            divisor (int): 输出通道数必须是此值的倍数。默认: 8。

        Returns:
            int: 调整后的通道数。
        """
        if not multiplier: # 如果乘数为0或None，则直接返回原宽度
            return width

        width *= multiplier # 应用乘数
        min_depth = min_depth or divisor # 设置最小值
        # 进行四舍五入并保证是divisor的倍数
        new_filters = max(min_depth,
                          int(width + divisor / 2) // divisor * divisor)
        # 如果结果过小，则增加divisor
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, multiplier):
        """
        根据深度乘数对重复次数（层数）进行四舍五入。

        Args:
            repeats (int): 原始重复次数。
            multiplier (float): 深度乘数 (gamma_d)。

        Returns:
            int: 调整后的重复次数。
        """
        if not multiplier: # 如果乘数为0或None，则直接返回原重复次数
            return repeats
        # 使用ceil向上取整，确保至少有一个块
        return int(math.ceil(multiplier * repeats))

    # 该模块由gamma_b参数化
    # 无时间步长
    def make_res_layer(self,
                       block,              # 要构建的残差块类型
                       layer_inplanes,     # 该res层输入特征的通道数
                       inplanes,           # 每个块输入特征的通道数，等于 base_channels * gamma_w
                       planes,             # 每个块瓶颈层输出特征的通道数，等于 base_channel * gamma_w * gamma_b
                       blocks,             # 残差块的数量
                       spatial_stride=1,   # 空间维度的步长
                       se_style='half',    # 插入SE模块的风格
                       se_ratio=None,      # SE单元的压缩比
                       use_swish=True,     # 是否使用Swish激活函数
                       norm_cfg=None,      # 归一化层配置
                       act_cfg=None,       # 激活函数层配置
                       conv_cfg=None,      # 卷积层配置
                       **kwargs):          # 其他参数
        """
        为ResNet3D构建一个残差层。

        Args:
            block (nn.Module): 要构建的残差模块。
            layer_inplanes (int): res层输入特征的通道数。
            inplanes (int): 每个块输入特征的通道数。
            planes (int): 每个块瓶颈层输出特征的通道数。
            blocks (int): 残差块的数量。
            spatial_stride (int): 残差块和卷积层中的空间步长。默认: 1。
            se_style (str): 将SE模块插入BlockX3D的方式，'half'表示插入一半的块，'all'表示插入所有块。默认: 'half'。
            se_ratio (float | None): Squeeze-and-Excitation单元的压缩比。如果设置为None，则表示不使用SE单元。默认: None。
            use_swish (bool): 是否在3x3x3卷积前后使用swish激活函数。默认: True。
            conv_cfg (dict | None): 卷积层的配置。默认: None。
            norm_cfg (dict | None): 归一化层的配置。默认: None。
            act_cfg (dict | None): 激活函数层的配置。默认: None。

        Returns:
            nn.Module: 为给定配置构建的残差层。
        """
        # 初始化下采样层为None
        downsample = None
        # 如果空间步长不是1或者输入输出通道数不同，则需要下采样层来匹配残差连接
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes,      # 下采样的输入通道数
                inplanes,            # 下采样的输出通道数
                kernel_size=1,       # 1x1x1卷积
                stride=(1, spatial_stride, spatial_stride), # 时间维度步长为1，空间维度为spatial_stride
                padding=0,           # 无填充
                bias=False,          # 无偏置
                conv_cfg=conv_cfg,   # 卷积配置
                norm_cfg=norm_cfg,   # 归一化配置
                act_cfg=None)        # 无激活函数

        # 初始化一个布尔列表，标记每个块是否使用SE模块
        use_se = [False] * blocks
        if self.se_style == 'all': # 如果风格是'all'，则所有块都使用SE
            use_se = [True] * blocks
        elif self.se_style == 'half': # 如果风格是'half'，则每隔一个块使用SE
            use_se = [i % 2 == 0 for i in range(blocks)]
        else:
            raise NotImplementedError # 如果风格未知，则抛出异常

        layers = [] # 存储该阶段所有块的列表
        # 添加第一个块，它可能需要下采样
        layers.append(
            block(
                layer_inplanes,    # 输入通道数
                planes,            # 瓶颈层通道数
                inplanes,          # 输出通道数
                spatial_stride=spatial_stride, # 空间步长
                downsample=downsample,         # 下采样层
                se_ratio=se_ratio if use_se[0] else None, # 根据use_se决定是否启用SE
                use_swish=use_swish,           # 是否使用Swish
                norm_cfg=norm_cfg,             # 归一化配置
                conv_cfg=conv_cfg,             # 卷积配置
                act_cfg=act_cfg,               # 激活函数配置
                **kwargs))                     # 其他参数

        # 添加剩余的块，它们不需要下采样，步长为1
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,        # 输入通道数（等于上一个块的输出）
                    planes,          # 瓶颈层通道数
                    inplanes,        # 输出通道数
                    spatial_stride=1,        # 空间步长为1
                    se_ratio=se_ratio if use_se[i] else None, # 根据use_se决定是否启用SE
                    use_swish=use_swish,       # 是否使用Swish
                    norm_cfg=norm_cfg,         # 归一化配置
                    conv_cfg=conv_cfg,         # 卷积配置
                    act_cfg=act_cfg,           # 激活函数配置
                    **kwargs))                # 其他参数

        # 将所有块组合成一个Sequential模块返回
        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """
        构建“茎”层，包括一个conv+norm+act模块和一个池化层。
        茎层是网络的入口，负责初步提取特征。
        """
        # 第一个卷积层：(1, 3, 3)，时间维度不压缩，空间维度减半
        self.conv1_s = ConvModule(
            self.in_channels,      # 输入通道数（如RGB为3）
            self.base_channels,    # 输出通道数
            kernel_size=(1, 3, 3), # 卷积核大小
            stride=(1, 2, 2),      # 步长，时间维1，空间维2
            padding=(0, 1, 1),     # 填充
            bias=False,            # 无偏置
            conv_cfg=self.conv_cfg,# 卷积配置
            norm_cfg=None,         # 无归一化
            act_cfg=None)          # 无激活函数
        # 第二个卷积层：(5, 1, 1)，时间维度压缩，空间维度不变
        # 使用分组卷积(groups=self.base_channels)，相当于对每个通道做1D卷积
        self.conv1_t = ConvModule(
            self.base_channels,    # 输入通道数
            self.base_channels,    # 输出通道数
            kernel_size=(5, 1, 1), # 卷积核大小
            stride=(1, 1, 1),      # 步长
            padding=(2, 0, 0),     # 填充
            groups=self.base_channels, # 分组卷积
            bias=False,            # 无偏置
            conv_cfg=self.conv_cfg,# 卷积配置
            norm_cfg=self.norm_cfg,# 归一化配置
            act_cfg=self.act_cfg)  # 激活函数配置

    def _freeze_stages(self):
        """冻结指定阶段之前的所有参数，防止其被优化。"""
        if self.frozen_stages >= 0: # 如果要冻结阶段0（茎层）
            self.conv1_s.eval() # 将层设置为评估模式
            self.conv1_t.eval()
            for param in self.conv1_s.parameters(): # 设置参数不参与梯度更新
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False

        # 循环冻结指定数量的阶段
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}') # 获取对应层
            m.eval() # 设置为评估模式
            for param in m.parameters(): # 设置参数不参与梯度更新
                param.requires_grad = False

    def init_weights(self):
        """
        初始化模型参数，可以从现有检查点加载，也可以从头开始。
        """
        for m in self.modules(): # 遍历所有子模块
            if isinstance(m, nn.Conv3d): # 如果是3D卷积层
                kaiming_init(m) # 使用Kaiming方法初始化权重
            elif isinstance(m, _BatchNorm): # 如果是批量归一化层
                constant_init(m, 1) # 将权重初始化为1

        # 如果启用零初始化残差
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BlockX3D): # 对于每个BlockX3D
                    # 将其最后一个卷积层的BN层权重初始化为0
                    # 这有助于在训练初期让残差块近似为恒等映射
                    constant_init(m.conv3.bn, 0)

        # 如果提供了预训练模型路径
        if isinstance(self.pretrained, str):
            logger = get_root_logger() # 获取日志记录器
            logger.info(f'从以下位置加载模型: {self.pretrained}') # 记录信息
            self.pretrained = cache_checkpoint(self.pretrained) # 缓存检查点
            # 加载预训练权重，strict=False允许部分参数不匹配
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        """
        定义前向传播的计算过程。

        Args:
            x (torch.Tensor): 输入数据。

        Returns:
            torch.Tensor: 骨干网络提取的输入样本特征。
        """
        x = self.conv1_s(x) # 通过第一个茎层卷积
        x = self.conv1_t(x) # 通过第二个茎层卷积
        # 依次通过各个阶段的残差层
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name) # 获取层对象
            x = res_layer(x) # 前向传播
        x = self.conv5(x) # 通过最后的卷积层
        return x # 返回最终特征

    def train(self, mode=True):
        """
        设置训练模式下的优化状态。

        Args:
            mode (bool): True表示进入训练模式，False表示评估模式。
        """
        super().train(mode) # 调用父类的train方法
        self._freeze_stages() # 冻结指定阶段
        if mode and self.norm_eval: # 如果是训练模式且要求BN层评估模式
            for m in self.modules(): # 遍历所有模块
                if isinstance(m, _BatchNorm): # 如果是BN层
                    m.eval() # 将其设置为评估模式（冻结运行时统计量）