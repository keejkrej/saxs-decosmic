"""
Comment: Single image processing
Dataclass: SingleConfig, SingleResult
Class: SingleProcessor
"""
from dataclasses import dataclass
import logging
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# =====================================================================
# Config and Result Dataclasses
# =====================================================================

@dataclass
class SingleConfig:
    """Configuration for SingleProcessor.
    
    Attributes:
        th_donut (int): Threshold for de-donut
        th_streak (int): Threshold for de-streak
        win_streak (int): Window size for de-streak
        exp_donut (int): Expansion of mask for de-donut
        exp_streak (int): Expansion of mask for de-streak
    """
    th_donut: int
    th_streak: int
    win_streak: int
    exp_donut: int
    exp_streak: int

@dataclass
class SingleResult:
    """Results of the image processing.
    
    Attributes:
        img_orig (np.ndarray): Input image
        img_half_clean (np.ndarray): Half-cleaned image after donut removal
        img_clean (np.ndarray): Cleaned image after both donut and streak removal
        mask_modifiable (np.ndarray): External mask for modifiable pixels, where True means can be modified
        mask_donut (np.ndarray): Mask for donut removal
        mask_streak (np.ndarray): Mask for streak removal
        mask_modifiable (np.ndarray): Combined modification mask
        sub_donut (np.ndarray): Subtracted donut artifacts
        sub_streak (np.ndarray): Subtracted streak artifacts
    """
    img_orig: np.ndarray
    img_half_clean: Optional[np.ndarray] = None
    img_clean: Optional[np.ndarray] = None
    mask_modifiable: np.ndarray
    mask_donut: Optional[np.ndarray] = None
    mask_streak: Optional[np.ndarray] = None
    mask_modified: Optional[np.ndarray] = None
    sub_donut: Optional[np.ndarray] = None
    sub_streak: Optional[np.ndarray] = None

# =====================================================================
# Single Image Processor Class
# =====================================================================

class SingleProcessor:
    """Processes a single image to remove high energy background.
    
    This class implements de-donut and de-streak algorithms for donut-shaped and streak-shaped features.
    
    Attributes:
        single_config (SingleConfig): Configuration for the single image processor
        result (SingleResult): Results of the single image processing
        shape (Tuple[int, int]): Shape of the image
        dtype (np.dtype): Data type of the image
    """

    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self,
                 img_orig: np.ndarray,
                 mask_modifiable: np.ndarray,
                 single_config: SingleConfig) -> None:
        """Initialize the image processor.
        
        Args:
            img: Input image data as numpy array
            mask: External mask for modifiable pixels, where True means can be modified
        """
        try:            
            if not isinstance(img_orig, np.ndarray):
                raise TypeError("Input image must be a numpy array")
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
        """Remove donut-shaped features.
        
        Args:
            img: Input image as numpy array
            mask: External mask for modifiable pixels, where True means can be modified
            
        Returns:
            Tuple containing:
                - Cleaned image with donut features removed
                - Modification mask indicating removed pixels
        """
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

            donut_mask = img_orig_copy > self.single_config.th_donut
            donut_mask_expanded = maximum_filter(donut_mask, size=self.single_config.exp_donut)
            mask_modified = donut_mask_expanded & mask_modifiable_copy
            img_orig_copy[mask_modified] = 0
            logger.debug(f"De-donut complete. Modified pixels: {np.sum(mask_modified)}")
            return img_orig_copy, mask_modified
        except Exception as e:
            logger.error(f"De-donut failed: {e}")
            raise
    
    def _de_streak(self, img_orig: np.ndarray, mask_modifiable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove streak-shaped features.
        
        Args:
            img: Input image as numpy array
            mask: External mask for modifiable pixels, where True means can be modified
            
        Returns:
            Tuple containing:
                - Cleaned image with streak features removed
                - Modification mask indicating removed pixels
        """
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
            streak_mask = img_conv > self.single_config.th_streak
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

    def clean_img(self) -> None:
        """Clean the image by removing high energy background.
        
        This method processes the image in two steps:
        1. Remove donut-shaped features
        2. Remove streak-shaped features
        
        The results are stored in the self.result object.
        """
        try:
            if self.single_result.img_orig is None or self.single_result.mask_modifiable is None:
                raise ValueError("Image and mask must be set before cleaning")
            
            logger.debug("Starting image cleaning process")

            self.single_result.img_half_clean, self.single_result.mask_donut = self._de_donut(self.single_result.img_orig, self.single_result.mask_modifiable)
            self.single_result.img_clean, self.single_result.mask_streak = self._de_streak(self.single_result.img_half_clean, self.single_result.mask_modifiable)
            
            self.single_result.mask_modified = self.single_result.mask_donut | self.single_result.mask_streak
            self.single_result.sub_donut = self.single_result.img_orig - self.single_result.img_half_clean
            self.single_result.sub_streak = self.single_result.img_half_clean - self.single_result.img_clean
            
            logger.debug(f"Image cleaning complete. Total modified pixels: {np.sum(self.single_result.mask_modified)}")
            logger.debug(f"Donut features removed: {np.sum(self.single_result.sub_donut)} photons, Streak features removed: {np.sum(self.single_result.sub_streak)} photons")
            return self.single_result
        except Exception as e:
            logger.error(f"Failed to clean image: {e}")
            raise