# Copyright (c) OpenMMLab. All rights reserved.
import io
import numpy as np
import os.path as osp
from mmcv.fileio import FileClient
from ..builder import PIPELINES
# 数据加载与解码模块，负责从视频文件中读取帧数据。 
# Decord 库高效读取视频文件。

# 初始化 Decord 视频读取器，为后续的视频帧解码做准备。这是视频处理流水线的第一步
@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        num_threads (int): Number of thread to decode the video. Default: 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None
    # 如果文件名没有扩展名，自动添加 .mp4
    # 场景: 数据集标注可能只提供基础路径
    def _get_videoreader(self, filename):
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')
# 创建文件客户端 支持不同的存储backend
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))# 读取视频为字节流
        container = decord.VideoReader(file_obj, num_threads=1)# 创建视频读取器，创建 Decord VideoReader 对象，返回值: Decord VideoReader 对象，可以用来解码视频帧
        return container

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'filename' not in results:
            assert 'frame_dir' in results
            results['filename'] = results['frame_dir'] + '.mp4'

        results['video_reader'] = self._get_videoreader(results['filename'])
        if 'total_frames' in results:
# 已有 total_frames: 验证与视频实际帧数是否一致
            assert results['total_frames'] == len(results['video_reader']), (
                'SkeFrames', results['total_frames'], 'VideoFrames', len(results['video_reader'])
            )
            # 没有 total_frames: 从视频读取器获取
        else:
            results['total_frames'] = len(results['video_reader'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str

# 使用 Decord 解码视频帧，根据帧索引提取指定的图像帧。这是视频处理流水线的第二步。
@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets. Default: 'accurate'.
    """
    # 精确解码，返回准确的帧	大多数场景（默认）， 快速解码，只返回关键帧即可，大规模场景识别数据集
    def __init__(self, mode='accurate'):
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container, frame_inds):
        if self.mode == 'accurate': # 精确解码，返回准确的帧（默认） 批量获取指定的索引的帧
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)# 重置到视频开头
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)# 获取下一帧
                frame = container.next() # 返回的是最近的关键帧，不是精确帧
                imgs.append(frame.asnumpy())
        return imgs
    # 大规模场景识别数据集（如 Kinetics）
    # 不需要精确时间定位
    # 速度优先
    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 从 results 中取出由 DecordInit 创建的视频读取器
        container = results['video_reader']
        # 保帧索引是一维 可能是多维数组（如多片段采样）
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        # 调用私有方法加载帧， imgs图像列表
        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container
        # 立即释放视频读取器，避免内存泄漏
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str

# ArrayDecode 支持 RGB 和光流两种模态
# 从预加载的 4D 数组中解码帧，不需要读取视频文件。适用于数据已经加载到内存的场景
@PIPELINES.register_module()
class ArrayDecode:
    """Load and decode frames with given indices from a 4D array.

    Required keys are "array and "frame_inds", added or modified keys are
    "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # modality: 'RGB' 或 'Flow'  array: 预加载的数组
        modality = results['modality']
        array = results['array']

        imgs = list()
        #  遍历帧索引并解码 
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for i, frame_idx in enumerate(results['frame_inds']):

            frame_idx += offset
            if modality == 'RGB':
                imgs.append(array[frame_idx])#直接从数组中取出帧，array[frame_idx] 形状: (H, W, C)
            elif modality == 'Flow':  # 光流有两个通道：x 方向和 y 方向
                imgs.extend(
                    [array[frame_idx, ..., 0], array[frame_idx, ..., 1]]) #：采样 8 帧光流 → 16 个图像（8 个 x + 8 个 y）
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
