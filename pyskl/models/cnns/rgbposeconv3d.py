import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm, print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d_slowfast import ResNet3dPathway


@BACKBONES.register_module()
class RGBPoseConv3D(nn.Module):
    """Slowfast backbone.

    Args:
        pretrained (str): The file path to a pretrained model.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 4.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 4.
    """
    def __init__(self,
                 pretrained=None,#预训练模型路径，支持从检查点加载
                 speed_ratio=4,#: 快慢路径时间维度比例
                 channel_ratio=4,#快路径通道数缩减比例
                 rgb_detach=False,
                 pose_detach=False,
                 rgb_drop_path=0,
                 pose_drop_path=0,
                 #双流路径配置策略
                 rgb_pathway=dict(
                    num_stages=4, # 4个阶段（标准ResNet结构）
                    lateral=True, # 启用横向连接
                    lateral_infl=1,
                    lateral_activate=(0, 0, 1, 1),# 在第3、4阶段激活横向连接
                    base_channels=64, # 基础通道数64
                    conv1_kernel=(1, 7, 7),
                    inflate=(0, 0, 1, 1)),
                 pose_pathway=dict(
                    num_stages=3,# 3个阶段（轻量化设计）
                    stage_blocks=(4, 6, 3),
                    lateral=True,# 启用横向链接
                    lateral_inv=True,
                    lateral_infl=16,
                    lateral_activate=(0, 1, 1),# 在第2、3阶段激活横向连接
                    in_channels=17, # 17个姿态关节点
                    base_channels=32, # 基础通道数32（更轻量）
                    out_indices=(2, ),
                    conv1_kernel=(1, 7, 7),
                    conv1_stride=(1, 1),
                    pool1_stride=(1, 1),
                    inflate=(0, 1, 1),
                    spatial_strides=(2, 2, 2),
                    temporal_strides=(1, 1, 1))):

        super().__init__()
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio#速度比例保存
        self.channel_ratio = channel_ratio#通道比例保存
        # 如果 RGB 通路有横向连接，添加速度和通道比例参数
        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio

        # 如果姿态通路有横向连接，添加速度和通道比例参数
        if pose_pathway['lateral']:
            pose_pathway['speed_ratio'] = speed_ratio
            pose_pathway['channel_ratio'] = channel_ratio
        # 构建 RGB 和姿态通路对象
        self.rgb_path = ResNet3dPathway(**rgb_pathway)
        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        # 保持丢弃的概率
        self.rgb_drop_path = rgb_drop_path 
        self.pose_drop_path = pose_drop_path
        # 权重初始化策略
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m) # Conv3D使用Kaiming初始化（适合RELU）
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)# BatchNorm初始化为1

        if isinstance(self.pretrained, str):# 如果有训练的路径的化记录即可
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger) # 构建与打印日志消息
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch seperately.
            self.rgb_path.init_weights()# 没有预训练的，直接进行初始化即可
            self.pose_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs, heatmap_imgs):
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.         #rgb图像
            heatmap_imgs (torch.Tensor): The input data. # 姿态热力图

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False
        # We assume base_channel for RGB and Pose are 64 and 32.
        # RGB 通路：第一层卷积
        x_rgb = self.rgb_path.conv1(imgs)
        # RGB 通路：最大池化
        x_rgb = self.rgb_path.maxpool(x_rgb)
        # N x 64 x 8 x 56 x 56
        # Pose 通路：第一层卷积
        x_pose = self.pose_path.conv1(heatmap_imgs)
        # Pose 通路：最大池化
        x_pose = self.pose_path.maxpool(x_pose)

        x_rgb = self.rgb_path.layer1(x_rgb)
        x_rgb = self.rgb_path.layer2(x_rgb)# RGB 通过第 2 层
        x_pose = self.pose_path.layer1(x_pose)
        
        # 多层横向融合: 在不同阶段交换和融合两种模态的特征
        
        # RGB 接收 Pose 特征（第一次横向连接）
        # 检查 RGB 通路是否有 layer2_lateral 模块
        if hasattr(self.rgb_path, 'layer2_lateral'):
            # 根据 rgb_detach 决定是否分离梯度
            feat = x_pose.detach() if self.rgb_detach else x_pose
            # 通过横向连接模块转换 Pose 特征
            x_pose_lateral = self.rgb_path.layer2_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        # 姿态通路接收 RGB 特征（第一次横向连接）
        # 检查姿态通路是否有 layer1_lateral 模块
        if hasattr(self.pose_path, 'layer1_lateral'):
            # 根据 pose_detach 决定是否分离梯度
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            # 通过横向连接模块转换 RGB 特征
            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            # 如果触发 DropPath，置零
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        # 检查 RGB 通路是否有 layer2_lateral 模块
        if hasattr(self.rgb_path, 'layer2_lateral'):
            # 将转换后的 Pose 特征拼接到 RGB 特征（dim=1 表示通道维度）
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)
        # 检查姿态通路是否有 layer1_lateral 模块
        if hasattr(self.pose_path, 'layer1_lateral'):
            # 将转换后的 RGB 特征拼接到 Pose 特征
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
            x_rgb = self.rgb_path.layer3(x_rgb)
        x_pose = self.pose_path.layer2(x_pose)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose
            x_pose_lateral = self.rgb_path.layer3_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer2_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer4(x_rgb)
        x_pose = self.pose_path.layer3(x_pose)

        return (x_rgb, x_pose)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self.training = True

    def eval(self):
        super().eval()
        self.training = False
