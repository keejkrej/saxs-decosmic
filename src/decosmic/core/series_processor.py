"""
Processor for handling series of XRD images.
"""
import os
import fabio
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List

from .processing_params import ProcessingParams
from .image_processor import ImageProcessor

class SeriesProcessor:
    """Processes a series of XRD images to remove cosmic ray artifacts."""
    
    def __init__(self, first_filename: str, user_mask: np.ndarray[bool] = True):
        self.first_filename = first_filename
        print(f"Loading {first_filename} ...")
        self.img_series = fabio.open_series(first_filename=self.first_filename)
        print(f"Loaded {self.img_series.nframes} images")
        self.img_num = self.img_series.nframes
        self.img_shape = self.img_series.get_frame(0).data.shape
        self.img_dtype = self.img_series.get_frame(0).data.dtype
        self.user_mask = user_mask
        
        # Processing parameters
        self.th_donut: Optional[int] = None
        self.th_mask: Optional[float] = None
        self.th_streak: Optional[int] = None
        self.win_streak: Optional[int] = None
        self.exp_donut: Optional[int] = None
        self.exp_streak: Optional[int] = None
        
        # Processing results
        self.img_avg: Optional[np.ndarray] = None
        self.img_binary_avg: Optional[np.ndarray] = None
        self.ring_mask: Optional[np.ndarray] = None
        self.combined_mask: Optional[np.ndarray] = None
        self.img_clean_avg: Optional[np.ndarray] = None
        self.sub_donut_avg: Optional[np.ndarray] = None
        self.sub_streak_avg: Optional[np.ndarray] = None
        self.img_diff_avg: Optional[np.ndarray] = None

    def load_params(self, params: ProcessingParams) -> None:
        """Load parameters from a ProcessingParams instance."""
        # Parameters are already validated by ProcessingParams
        self.th_donut = int(params.th_donut)
        self.th_mask = float(params.th_mask)
        self.th_streak = int(params.th_streak)
        self.win_streak = int(params.win_streak)
        self.exp_donut = int(params.exp_donut)
        self.exp_streak = int(params.exp_streak)

    def get_img(self, idx: int) -> np.ndarray:
        """Get a single image from the series."""
        img = self.img_series.get_frame(idx).data.astype(np.int32)
        img = np.nan_to_num(img, nan=-1)
        img = np.clip(img, 0, None)
        return img
    
    def avg_img(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Calculate average image and binary average."""
        img_sum = np.zeros(self.img_shape, dtype=np.float64)
        img_binary_sum = np.zeros(self.img_shape, dtype=np.float64)
        print('Averaging images ...')
        
        for i in range(self.img_num):
            img = self.get_img(i)
            img_sum += img
            img_binary = img > 0
            img_binary_sum += img_binary
            if progress_callback:
                progress_callback(i)
        
        self.img_avg = img_sum / self.img_num
        self.img_binary_avg = img_binary_sum / self.img_num

    def mask_img(self) -> None:
        """Create mask for ring artifacts."""
        self.ring_mask = self.img_binary_avg < self.th_mask
        self.combined_mask = self.ring_mask & self.user_mask

    def avg_clean_img(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Process all images to remove cosmic ray artifacts."""
        self.avg_img(progress_callback)
        self.mask_img()
        
        img_clean_sum = np.zeros(self.img_shape, dtype=np.float64)
        img_clean_num = np.ones(self.img_shape, dtype=np.int32) * self.img_num
        sub_donut_sum = np.zeros(self.img_shape, dtype=np.float64)
        sub_streak_sum = np.zeros(self.img_shape, dtype=np.float64)
        
        print('Cleaning images ...')
        for i in range(self.img_num):
            img = self.get_img(i)
            processor = ImageProcessor(img, self.combined_mask)
            processor.load_params(
                self.th_donut, self.th_streak, self.win_streak,
                self.exp_donut, self.exp_streak
            )
            processor.clean_img()
            
            img_clean_sum += processor.img_clean
            sub_donut_sum += processor.sub_donut
            sub_streak_sum += processor.sub_streak
            img_clean_num -= processor.mod_mask
            
            if progress_callback:
                progress_callback(i)
        
        self.img_clean_avg = img_clean_sum / img_clean_num
        self.sub_donut_avg = sub_donut_sum / self.img_num
        self.sub_streak_avg = sub_streak_sum / self.img_num
        self.img_diff_avg = self.img_avg - self.img_clean_avg

    def save_results(self, output_dir: str) -> None:
        """Save processing results as TIFF files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save average images
        fabio.tifimage.tifimage(data=self.img_avg).write(
            os.path.join(output_dir, 'avg.tif')
        )
        fabio.tifimage.tifimage(data=self.img_clean_avg).write(
            os.path.join(output_dir, 'clean.tif')
        )
        fabio.tifimage.tifimage(data=self.img_diff_avg).write(
            os.path.join(output_dir, 'diff.tif')
        )
        
        # Save masks
        fabio.tifimage.tifimage(data=self.ring_mask.astype(np.int32)).write(
            os.path.join(output_dir, 'mask.tif')
        )
        
        # Save subtracted components
        fabio.tifimage.tifimage(data=self.sub_donut_avg).write(
            os.path.join(output_dir, 'donut.tif')
        )
        fabio.tifimage.tifimage(data=self.sub_streak_avg).write(
            os.path.join(output_dir, 'streak.tif')
        ) 