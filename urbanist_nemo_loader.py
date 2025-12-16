import torch
import numpy as np
import rasterio
from pathlib import Path
from torchdata.datapipes.iter import IterDataPipe


class UrbanistNeMoDataPipe(IterDataPipe):
    """
    Minimal Physics-NeMo data loader stub

    Reads:
      - dem.tif  -> features
      - HRF.tif  -> labels (flood raster)

    Returns:
      features : Tensor (1, H, W)
      labels   : Tensor (1, H, W)
    """

    def __init__(
        self,
        data_dir,
        dem_range=(0.0, 600.0),
        hrf_range=(0.0, 6.0),
        dtype=torch.float32,
    ):
        self.data_dir = Path(data_dir)

        self.dem_path = self.data_dir / "dem.tif"
        self.hrf_path = self.data_dir / "HRF.tif"

        if not self.dem_path.exists():
            raise FileNotFoundError("dem.tif not found in data_dir")

        if not self.hrf_path.exists():
            raise FileNotFoundError("HRF.tif not found in data_dir")

        self.dem_min, self.dem_max = dem_range
        self.hrf_min, self.hrf_max = hrf_range
        self.dtype = dtype

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = 0.0
        return arr

    def _minmax(self, x, lo, hi):
        return (x - lo) / (hi - lo + 1e-6)

    def __iter__(self):
        dem = self._read_raster(self.dem_path)
        hrf = self._read_raster(self.hrf_path)

        dem = self._minmax(dem, self.dem_min, self.dem_max)
        hrf = self._minmax(hrf, self.hrf_min, self.hrf_max)

        features = torch.tensor(dem, dtype=self.dtype).unsqueeze(0)
        labels = torch.tensor(hrf, dtype=self.dtype).unsqueeze(0)

        yield {
            "features": features,
            "labels": labels
        }
