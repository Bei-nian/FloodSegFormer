import os.path as osp
import mmcv
import numpy as np
from PIL import Image
from mmseg.registry import DATASETS  
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class AIFloodSenseDataset(BaseSegDataset):
    """AIFloodSense dataset for flood scene semantic segmentation.

    4 classes after mask conversion:
        0 - background
        1 - flood
        2 - sky
        3 - building
    Images are .jpg (1024x768), converted masks are .png with class indices.
    """

    METAINFO = {
        'classes': ('background', 'flood', 'sky', 'building'),
        'palette': [
            [0, 0, 0],          # background - 黑
            [0, 100, 220],      # flood      - 蓝
            [135, 206, 235],    # sky        - 天蓝
            [200, 80, 60],      # building   - 砖红
        ]
    }

    def __init__(self, **kwargs):
        super(AIFloodSenseDataset, self).__init__(
            img_suffix='.jpg',           
            seg_map_suffix='.png',      
            reduce_zero_label=False,     
            **kwargs)

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