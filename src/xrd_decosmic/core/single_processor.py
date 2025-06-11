"""Single image processing module with SingleConfig, SingleResult dataclasses and SingleProcessor class."""
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from typing import Tuple

import tifffile

logger = logging.getLogger(__name__)

# =====================================================================
# Config and Result Dataclasses
# =====================================================================

@dataclass
class SingleConfig:
    """Configuration parameters for SingleProcessor."""
    th_donut: int
    th_streak: int
    win_streak: int
    exp_donut: int
    exp_streak: int

@dataclass
class SingleResult:
    """Results container for single image processing with original, cleaned images and masks."""
    img_orig: np.ndarray
    img_half_clean: np.ndarray | None = None
    img_clean: np.ndarray | None = None
    mask_modifiable: np.ndarray | None = None
    mask_donut: np.ndarray | None = None
    mask_streak: np.ndarray | None = None
    mask_combined: np.ndarray | None = None
    sub_donut: np.ndarray | None = None
    sub_streak: np.ndarray | None = None

    def save(self, output_dir: str, prefix: str = '') -> None:
        """Save all result arrays as TIFF files in the specified directory."""
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in self.__dict__.items():
            if value is not None:
                tifffile.imwrite(
                    output_path / f'{prefix}_{key}.tif',
                    value
                )

    def load(self, input_dir: str, prefix: str = '') -> None:
        """Load all result arrays from TIFF files in the specified directory."""
        input_path = Path(input_dir).resolve()
        for key in self.__dict__:
            file_path = input_path / f'{prefix}_{key}.tif'
            if file_path.exists():
                setattr(self, key, tifffile.imread(file_path))
            else:
                raise FileNotFoundError(f"File {file_path} does not exist")

# =====================================================================
# Single Image Processor Class
# =====================================================================

class SingleProcessor:
    """Processes single images to remove high energy background using de-donut and de-streak algorithms."""

    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self,
                 img_orig: np.ndarray,
                 single_config: SingleConfig,
                 mask_modifiable: np.ndarray | None = None) -> None:
        """Initialize the image processor with input image, configuration and optional mask."""
        try:
            if not isinstance(img_orig, np.ndarray):
                raise TypeError("Input image must be a numpy array")
            if mask_modifiable is None:
                mask_modifiable = np.ones(img_orig.shape, dtype=bool)
            if not isinstance(mask_modifiable, np.ndarray):
                raise TypeError("Input mask must be a numpy array")
            if img_orig.shape != mask_modifiable.shape:
                raise ValueError(f"Image shape {img_orig.shape} does not match mask shape {mask_modifiable.shape}")
            
            self.single_result = SingleResult(
                img_orig=img_orig,
                mask_modifiable=mask_modifiable,
            )
            self.single_config = single_config
            self.shape = img_orig.shape
            self.dtype = img_orig.dtype
            
            logger.debug(f"SingleProcessor initialized.")
            logger.debug(f"Configuration: {self.single_config}")
        except Exception as e:
            logger.error(f"Failed to initialize SingleProcessor: {e}")
            raise

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _de_donut(self, img_orig: np.ndarray, mask_modifiable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove donut-shaped features using threshold-based detection and morphological expansion."""
        try:
            if self.single_config is None:
                raise ValueError("Configuration is not set")
            if not isinstance(img_orig, np.ndarray) or not isinstance(mask_modifiable, np.ndarray):
                raise TypeError("Input image and mask must be numpy arrays")
            if img_orig.shape != mask_modifiable.shape:
                raise ValueError(f"Image shape {img_orig.shape} does not match mask shape {mask_modifiable.shape}")
            
            logger.debug(f"Starting de-donut with threshold: {self.single_config.th_donut}")

            img_orig_copy = np.copy(img_orig)
            mask_modifiable_copy = np.copy(mask_modifiable)

            donut_mask = img_orig_copy >= self.single_config.th_donut
            donut_mask_expanded = maximum_filter(donut_mask, size=self.single_config.exp_donut)
            mask_modified = donut_mask_expanded & mask_modifiable_copy
            img_orig_copy[mask_modified] = 0
            logger.debug(f"De-donut complete. Modified pixels: {np.sum(mask_modified)}")
            return img_orig_copy, mask_modified
        except Exception as e:
            logger.error(f"De-donut failed: {e}")
            raise
    
    def _de_streak(self, img_orig: np.ndarray, mask_modifiable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove streak-shaped features using convolution-based detection and morphological expansion."""
        try:
            if self.single_config is None:
                raise ValueError("Configuration is not set")
            if not isinstance(img_orig, np.ndarray) or not isinstance(mask_modifiable, np.ndarray):
                raise TypeError("Input image and mask must be numpy arrays")
            if img_orig.shape != mask_modifiable.shape:
                raise ValueError(f"Image shape {img_orig.shape} does not match mask shape {mask_modifiable.shape}")
            
            logger.debug(f"Starting de-streak with threshold: {self.single_config.th_streak} and window size: {self.single_config.win_streak}")

            img_orig_copy = np.copy(img_orig)
            mask_modifiable_copy = np.copy(mask_modifiable)
            
            img_binary = (img_orig_copy > 0).astype(np.int32)
            img_binary = img_binary * mask_modifiable_copy
            conv_kernel = np.ones((self.single_config.win_streak, self.single_config.win_streak), dtype=self.dtype)
            img_conv = convolve(img_binary, conv_kernel, mode='constant', cval=0)
            img_conv = img_conv * img_binary
            streak_mask = img_conv >= self.single_config.th_streak
            streak_mask_expanded = maximum_filter(streak_mask, size=self.single_config.exp_streak)
            mask_modified = streak_mask_expanded & mask_modifiable_copy
            img_orig_copy[mask_modified] = 0
            logger.debug(f"De-streak complete. Modified pixels: {np.sum(mask_modified)}")
            return img_orig_copy, mask_modified
        except Exception as e:
            logger.error(f"De-streak failed: {e}")
            raise

    # =====================================================================
    # Public Methods
    # =====================================================================

    def clean_img(self) -> SingleResult:
        """Clean the image by sequentially removing donut-shaped and streak-shaped features."""
        try:
            if self.single_result.img_orig is None or self.single_result.mask_modifiable is None:
                raise ValueError("Image and mask must be set before cleaning")
            
            logger.debug("Starting image cleaning process")

            self.single_result.img_half_clean, self.single_result.mask_donut = self._de_donut(self.single_result.img_orig, self.single_result.mask_modifiable)
            self.single_result.img_clean, self.single_result.mask_streak = self._de_streak(self.single_result.img_half_clean, self.single_result.mask_modifiable)
            
            self.single_result.mask_combined = self.single_result.mask_donut | self.single_result.mask_streak
            self.single_result.sub_donut = self.single_result.img_orig - self.single_result.img_half_clean
            self.single_result.sub_streak = self.single_result.img_half_clean - self.single_result.img_clean
            
            logger.debug("Image cleaning process completed successfully")
            return self.single_result
        except Exception as e:
            logger.error(f"Image cleaning failed: {e}")
            raise