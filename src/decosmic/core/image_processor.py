"""
Image processor for removing cosmic rays from XRD images.
"""
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from typing import Tuple, Optional

class ImageProcessor:
    """Processes XRD images to remove cosmic ray artifacts."""

    def __init__(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]):
        """
        Initialize the image processor.
        
        Args:
            img: Input image data
            mask: Mask for valid pixels
        """
        self.img = img
        self.mask = mask
        
        # Parameters
        self.th_donut: int = 0
        self.th_streak: int = 0
        self.win_streak: int = 0
        self.exp_donut: int = 0
        self.exp_streak: int = 0
        
        # Results
        self.img_clean: Optional[np.ndarray] = None
        self.img_intermediate: Optional[np.ndarray] = None
        self.mod_mask: Optional[np.ndarray] = None
        self.sub_donut: Optional[np.ndarray] = None
        self.sub_streak: Optional[np.ndarray] = None

    def load_params(self, th_donut: int, th_streak: int, win_streak: int,
                   exp_donut: int = 9, exp_streak: int = 3) -> None:
        """
        Load processing parameters.
        
        Args:
            th_donut: Threshold for donut detection
            th_streak: Threshold for streak detection
            win_streak: Window size for streak detection
            exp_donut: Exponent for donut detection
            exp_streak: Exponent for streak detection
        """
        self.th_donut = th_donut
        self.th_streak = th_streak
        self.win_streak = win_streak
        self.exp_donut = exp_donut
        self.exp_streak = exp_streak

    def de_donut(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]) -> Tuple[np.ndarray[np.int32], np.ndarray[bool]]:
        """
        Remove donut-shaped cosmic ray artifacts.
        
        Args:
            img: Input image
            mask: Valid pixel mask
            
        Returns:
            Tuple of (cleaned image, modification mask)
        """
        img_copy = np.copy(img)
        donut_mask = img_copy > self.th_donut
        donut_mask_expanded = maximum_filter(donut_mask, size=self.exp_donut)
        mod_mask = donut_mask_expanded & mask
        img_copy[mod_mask] = 0
        return img_copy, mod_mask
    
    def de_streak(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]) -> Tuple[np.ndarray[np.int32], np.ndarray[bool]]:
        """
        Remove streak-shaped cosmic ray artifacts.
        
        Args:
            img: Input image
            mask: Valid pixel mask
            
        Returns:
            Tuple of (cleaned image, modification mask)
        """
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
        return img_copy, mod_mask
    
    def clean_img(self) -> None:
        """Clean the image by removing cosmic ray artifacts."""
        img_copy = np.copy(self.img)
        img_donut, mask_donut = self.de_donut(img_copy, self.mask)
        self.img_intermediate = np.copy(img_donut)
        img_streak, mask_streak = self.de_streak(img_donut, self.mask)
        self.img_clean = np.copy(img_streak)
        self.mod_mask = mask_donut | mask_streak
        self.sub_donut = img_copy - img_donut
        self.sub_streak = img_donut - img_streak
