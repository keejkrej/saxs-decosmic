"""
Image processor for removing cosmic rays from XRD images.
This module provides functionality for detecting and removing cosmic ray artifacts
from XRD images, including both donut-shaped and streak-shaped artifacts.
"""
import logging
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from typing import Tuple
logger = logging.getLogger(__name__)
# =====================================================================
# Image Processor Class
# =====================================================================
class ImageProcessor:
    """Processes XRD images to remove cosmic ray artifacts.
    This class implements algorithms for detecting and removing cosmic ray
    artifacts from XRD images. It supports both donut-shaped and streak-shaped
    artifact removal with configurable parameters.
    Attributes:
        img (np.ndarray): Input image data
        mask (np.ndarray): Mask for valid pixels, where True means can be modified
        th_donut (int): Threshold for donut detection
        th_streak (int): Threshold for streak detection
        win_streak (int): Window size for streak detection
        exp_donut (int): Expansion of donut mask
        exp_streak (int): Expansion of streak mask
        img_clean (np.ndarray): Cleaned image after processing
        img_intermediate (np.ndarray): Intermediate image after donut removal
        mod_mask (np.ndarray): Combined modification mask
        sub_donut (np.ndarray): Subtracted donut artifacts
        sub_streak (np.ndarray): Subtracted streak artifacts
    """
    # =====================================================================
    # Initialization
    # =====================================================================
    def __init__(self,
                 img: np.ndarray,
                 mask: np.ndarray,
                 th_donut: int,
                 th_streak: int,
                 win_streak: int,
                 exp_donut: int,
                 exp_streak: int) -> None:
        """Initialize the image processor.
        Args:
            img: Input image data as numpy array
            mask: Mask for valid pixels, where True means can be modified
        """
        try:
            logger.debug("Initializing ImageProcessor with parameters: th_donut=%d, th_streak=%d, win_streak=%d, exp_donut=%d, exp_streak=%d",
                        th_donut, th_streak, win_streak, exp_donut, exp_streak)
            if not isinstance(img, np.ndarray):
                raise TypeError("Input image must be a numpy array")
            if not isinstance(mask, np.ndarray):
                raise TypeError("Input mask must be a numpy array")
            if img.shape != mask.shape:
                raise ValueError(f"Image shape {img.shape} does not match mask shape {mask.shape}")
            self.img = img
            self.mask = mask
            # Parameters
            self.th_donut = th_donut
            self.th_streak = th_streak
            self.win_streak = win_streak
            self.exp_donut = exp_donut
            self.exp_streak = exp_streak
            # Results
            self.img_intermediate = None
            self.img_clean = None
            self.mask_donut = None
            self.mask_streak = None
            self.mod_mask = None
            self.sub_donut = None
            self.sub_streak = None
            logger.debug("ImageProcessor initialized with image shape: %s", img.shape)
        except Exception as e:
            logger.error(f"Failed to initialize ImageProcessor: {str(e)}")
            raise
    # =====================================================================
    # Private Methods
    # =====================================================================
    def _de_donut(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove donut-shaped cosmic ray artifacts.
        Args:
            img: Input image as numpy array
            mask: Valid pixel mask as boolean array
        Returns:
            Tuple containing:
                - Cleaned image with donut artifacts removed
                - Modification mask indicating removed pixels
        """
        try:
            logger.debug("Starting donut artifact removal with threshold: %d", self.th_donut)
            if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
                raise TypeError("Input image and mask must be numpy arrays")
            if img.shape != mask.shape:
                raise ValueError(f"Image shape {img.shape} does not match mask shape {mask.shape}")
            img_copy = np.copy(img)
            donut_mask = img_copy > self.th_donut
            donut_mask_expanded = maximum_filter(donut_mask, size=self.exp_donut)
            mod_mask = donut_mask_expanded & mask
            img_copy[mod_mask] = 0
            logger.debug("Donut removal complete. Modified pixels: %d", np.sum(mod_mask))
            return img_copy, mod_mask
        except Exception as e:
            logger.error(f"Failed to remove donut artifacts: {str(e)}")
            raise
    def _de_streak(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove streak-shaped cosmic ray artifacts.
        Args:
            img: Input image as numpy array
            mask: Valid pixel mask as boolean array
        Returns:
            Tuple containing:
                - Cleaned image with streak artifacts removed
                - Modification mask indicating removed pixels
        """
        try:
            logger.debug("Starting streak artifact removal with threshold: %d and window size: %d", 
                        self.th_streak, self.win_streak)
            if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
                raise TypeError("Input image and mask must be numpy arrays")
            if img.shape != mask.shape:
                raise ValueError(f"Image shape {img.shape} does not match mask shape {mask.shape}")
            img_copy = np.copy(img)
            img_binary = (img_copy > 0).astype(np.int32)
            img_binary = img_binary * mask
            conv_kernel = np.ones((self.win_streak, self.win_streak), dtype=np.int32)
            img_conv = convolve(img_binary, conv_kernel, mode='constant', cval=0)
            img_conv = img_conv * img_binary
            streak_mask = img_conv > self.th_streak
            streak_mask_expanded = maximum_filter(streak_mask, size=self.exp_streak)
            mod_mask = streak_mask_expanded & mask
            img_copy[mod_mask] = 0
            logger.debug("Streak removal complete. Modified pixels: %d", np.sum(mod_mask))
            return img_copy, mod_mask
        except Exception as e:
            logger.error(f"Failed to remove streak artifacts: {str(e)}")
            raise
    # =====================================================================
    # Public Methods
    # =====================================================================
    def clean_img(self) -> None:
        """Clean the image by removing cosmic ray artifacts.
        This method processes the image in two steps:
        1. Remove donut-shaped artifacts
        2. Remove streak-shaped artifacts
        The results are stored in the instance attributes:
        - img_clean: Final cleaned image
        - img_intermediate: Image after donut removal
        - mod_mask: Combined modification mask
        - sub_donut: Subtracted donut artifacts
        - sub_streak: Subtracted streak artifacts
        - mask_donut: Mask for donut removal
        - mask_streak: Mask for streak removal
        """
        try:
            logger.debug("Starting image cleaning process")
            if self.img is None or self.mask is None:
                raise ValueError("Image and mask must be set before cleaning")
            img_copy = np.copy(self.img)
            img_donut, self.mask_donut = self._de_donut(img_copy, self.mask)
            img_streak, self.mask_streak = self._de_streak(img_donut, self.mask)
            self.img_intermediate = img_donut
            self.img_clean = img_streak
            self.mod_mask = self.mask_donut | self.mask_streak
            self.sub_donut = img_copy - img_donut
            self.sub_streak = img_donut - img_streak
            logger.debug("Image cleaning complete. Total modified pixels: %d", np.sum(self.mod_mask))
            logger.debug("Donut artifacts removed: %d photons, Streak artifacts removed: %d photons",
                        np.sum(self.sub_donut), np.sum(self.sub_streak))
        except Exception as e:
            logger.error(f"Failed to clean image: {str(e)}")
            raise