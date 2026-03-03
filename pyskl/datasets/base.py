# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import copy
import mmcv
import numpy as np
import os.path as osp
import torch
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from torch.utils.data import Dataset

from pyskl.smp import auto_mix2
from ..core import mean_average_precision, mean_class_accuracy, top_k_accuracy
from .pipelines import Compose

#视频动作识别数据集的抽象基类 BaseDataset。它遵循抽象工厂模式，为所有具体的视频数据集提供了统一的接口和一些通用的功能实现。
class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held. Default: ''.
        test_mode (bool): Store True when building test or validation dataset. Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of different filename format. However,
            if taking videos as input, it should be set to 0, since frames loaded from videos count from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'. Default: 'RGB'.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
    """

    def __init__(self,
                 ann_file, #标注文件路径
                 pipeline,#预处理的流水线
                 data_prefix='',#前缀
                 test_mode=False,#是否测试模式
                 multi_class=False,
                 num_classes=None,#分类类别
                 start_index=1,
                 modality='RGB',#RGB、Flow、Audio等模态可以挑选  这里选择RGB
                 memcached=False, #是否使用memcached缓存
                 mc_cfg=('localhost', 22077)):#memcached的配置
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        # Note: Currently, memcached only works for PoseDataset 缓存
        self.memcached = memcached
        self.mc_cfg = mc_cfg
        self.cli = None
        # 数据处理管道 转换为可执行的compose对象
        self.pipeline = Compose(pipeline) 
        # 加载标注文件，获取视频信息
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        #加载 JSON 格式注释文件的通用实现。
        # 它读取 JSON 文件，然后将视频文件路径（frame_dir 或 filename）与 data_prefix 拼接，以形成完整的路径
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos
    #将 video_infos 按照标签 (label) 进行分组，返回一个字典，键是标签索引，值是属于该标签的视频信息列表。这在某些需要按类别采样的场景下很有用
    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr
    #模型性能评估的核心方法
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 #评估指标的方式 top_k_accuracy、mean_class_accuracy、mean_average_precision多个选择
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # 检查结果是否为列表
        # 检查结果列表的长度是否与数据集长度相同
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')
        # 情况1: 嵌套列表/元组 处理多个模型输出的情况 
        # 对每个输出单独评估，结果合并为 {metric_name}_{index}: value
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            num_results = len(results[0])
            eval_results = dict()
            for i in range(num_results):
                eval_results_cur = self.evaluate(
                    [x[i] for x in results], metrics, metric_options, logger, **deprecated_kwargs)
                eval_results.update({f'{k}_{i}': v for k, v in eval_results_cur.items()})
            return eval_results
        # 情况2: 字典格式 
        # 处理多模态或多任务输出的情况
        # 对每个键对应的结果单独评估
        # 特别支持RGBPoseConv3D模型 
        # 检测到同时包含 'rgb' 和 'pose' 键的结果
        # 使用 auto_mix2 函数自动混合两种模态的预测结果
        # 生成组合评估结果 RGBPose_{mix_type}_{metric}
        elif isinstance(results[0], dict):
            eval_results = dict()
            for key in results[0]:
                results_cur = [x[key] for x in results]
                eval_results_cur = self.evaluate(results_cur, metrics, metric_options, logger, **deprecated_kwargs)
                eval_results.update({f'{key}_{k}': v for k, v in eval_results_cur.items()})
            # Ad-hoc for RGBPoseConv3D
            if len(results[0]) == 2 and 'rgb' in results[0] and 'pose' in results[0]:
                rgb = [x['rgb'] for x in results]
                pose = [x['pose'] for x in results]
                preds = auto_mix2([rgb, pose])
                for k in preds:
                    eval_results_cur = self.evaluate(preds[k], metrics, metric_options, logger, **deprecated_kwargs)
                    eval_results.update({f'RGBPose_{k}_{key}': v for key, v in eval_results_cur.items()})

            return eval_results

        # Protect ``metric_options`` since it uses mutable value as default
        # 深拷贝指标配置以避免修改原始参数
        metric_options = copy.deepcopy(metric_options)
        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)
        #标准化指标参数为列表格式以及检查指标是否支持
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]
        #默认计算Top-1至Top-5准确率
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            #平均类别准确率
            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue
            #平均精度均值 map
            if metric == 'mean_average_precision':
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                mAP = mean_average_precision(results, gt_labels_arrays)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results
    # 关键特性
    # 1. 灵活的结果格式支持
    # 单模型输出: [pred1, pred2, ...]
    # 多模型输出: [[pred1a, pred1b], [pred2a, pred2b], ...]
    # 多模态输出: [{'rgb': pred1, 'pose': pred1}, ...]
    
    
    
    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)
    # 工作流程总结
    # 数据获取 → 从视频信息列表获取指定索引的样本
    # 缓存检查 → 如果启用缓存，尝试从memcached获取数据
    # 缓存处理 → 缓存命中直接使用，未命中则从文件加载并更新缓存
    # 元数据添加 → 添加数据类型、帧索引等信息
    # 标签编码 → 多分类任务时转换为one-hot编码
    # 流水线处理 → 执行数据增强和预处理操作
    # 返回结果 → 返回处理完成的样本数据
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # 内存缓存的处理
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]
        #添加数据类型信息（RGB/Flow/Audio）
        #设置帧起始索引
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        #onehot 数组从 [0., 0., 0., ..., 0.] 变成了 [0., 0., 0., 0., 0., 1., 0., ..., 0.]
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes) #类别数的总量
            onehot[results['label']] = 1.
            results['label'] = onehot  #让视频当前的分类标签变成onehot编码的形式

        results['test_mode'] = self.test_mode
        return self.pipeline(results)#流水线处理 → 执行数据增强和预处理操作

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]

        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        return self.prepare_test_frames(idx) if self.test_mode else self.prepare_train_frames(idx)
