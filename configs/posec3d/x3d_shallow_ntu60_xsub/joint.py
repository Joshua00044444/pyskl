model = dict(
    type='Recognizer3D',# 3D动作识别器基类
    backbone=dict(
        type='X3D',#主干网络：采用 X3D 架构（一种高效的 3D CNN）17个关节点 输入数据：处理人体关节点（17个关节点）的热力图
        gamma_d=1,# 时间维度扩展因子 X3D的参数而已
        in_channels=17,# 17个关节点 输入数据：处理人体关节点（17个关节点）的热力图
        base_channels=24,# 基础通道数
        num_stages=3,# 网络阶段数：3个下采样阶段
        se_ratio=None,# Squeeze-and-Excitation模块比例（未使用）
        use_swish=False,# 是否使用Swish激活函数
        stage_blocks=(2, 5, 3),# 各阶段残差块数配置
        spatial_strides=(2, 2, 2)),#渐进式下采样，逐步减少空间分辨率，增大感受野去
    cls_head=dict(
        type='I3DHead',#分类头：使用 I3DHead 进行动作分类
        in_channels=216, # 最终特征通道数 ，该数是X3D主干网络最后一个阶段输出的特征通道
        num_classes=60, # NTU60数据集的60个动作类别
        dropout=0.5), # 防止过拟合
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset' 
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'# 数据集标注文件路径
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]# 左侧关节点索引
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]# 右侧关节点索引
train_pipeline = [ #训练流水线
    dict(type='UniformSampleFrames', clip_len=48),# 时间采样，统一视频片段48帧长度
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),# 随机裁剪增强
    dict(type='Resize', scale=(56, 56), keep_ratio=False), # 固定尺寸到56x56
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),# 水平翻转增强
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),# 生成关节点热力图
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),# 格式化为NCTHW张量
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),# 数据收集
    dict(type='ToTensor', keys=['imgs', 'label'])# 转换为PyTorch张量
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32, # 每个GPU的批次大小
    workers_per_gpu=4,# 每个GPU的数据加载进程数
    test_dataloader=dict(videos_per_gpu=1),# 测试时批次大小为1
    train=dict(
        type='RepeatDataset',  # 重复数据集增强
        times=10,# 重复10次
        dataset=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))# 梯度裁剪
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0) # 余弦退火学习率
total_epochs = 24 # 总训练轮数
checkpoint_config = dict(interval=1) # 每轮保存检查点
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5)) # 评估指标
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # 日志配置
log_level = 'INFO' # 日志配置
work_dir = './work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint' # 工作目录
