import os.path as osp
import mmcv
import numpy as np
from PIL import Image
from mmseg.registry import DATASETS  
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FloodNetDataset(BaseSegDataset):
    """FloodNet 洪灾场景语义分割数据集。

    10 个类别（FloodNet Challenge 官方定义）：
        0 - background
        1 - building-flooded
        2 - building-non-flooded
        3 - road-flooded
        4 - road-non-flooded
        5 - water
        6 - tree
        7 - vehicle
        8 - pool
        9 - grass

    """

    METAINFO = dict(
        classes=(
            # 'background',
            'building-flooded',
            'building-non-flooded',
            'road-flooded',
            'road-non-flooded',
            'water',
            'tree',
            'vehicle',
            'pool',
            'grass',
        ),
        palette=[
            # [0,   0,   0  ],   # 0  background         黑色
            [220, 20,  60 ],   # 1  building-flooded    深红
            [197, 235, 19 ],   # 2  building-non-flooded 黄绿
            [128, 64,  128],   # 3  road-flooded        紫色
            [105, 105, 105],   # 4  road-non-flooded    灰色
            [0,   191, 255],   # 5  water               亮蓝
            [34,  139, 34 ],   # 6  tree                深绿
            [255, 165, 0  ],   # 7  vehicle             橙色
            [0,   128, 128],   # 8  pool                深青
            [124, 252, 0  ],   # 9  grass               浅绿
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,   
            **kwargs
        )

    def results2img(self, results, imgfile_prefix, indices=None):
        """Save predictions as palette PNGs for visualization."""
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        if indices is None:
            indices = list(range(len(self)))

        prog_bar = mmcv.ProgressBar(len(indices))
        for idx in indices:
            result = results[idx]
            data_info = self.get_data_info(idx)
            filename = data_info['img_path']
            basename = osp.splitext(osp.basename(filename))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            palette = np.array(self.METAINFO['palette'], dtype=np.uint8)
            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, indices=None):
        import tempfile
        assert isinstance(results, list), 'results must be a list'
        if indices is None:
            indices = list(range(len(self)))
        assert len(results) == len(self)

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, indices)
        return result_files, tmp_dir