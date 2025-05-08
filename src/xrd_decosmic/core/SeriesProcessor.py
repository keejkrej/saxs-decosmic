"""
Processor for handling series of XRD images.

This module provides functionality for processing series of XRD images to remove
cosmic ray artifacts. It supports both single image processing and batch processing
of entire series.
"""
import json
import logging
import os
import fabio
import numpy as np
from typing import Optional
from pathlib import Path

from tqdm import tqdm

from .ImageProcessor import ImageProcessor
from .ImageSeries import ImageSeries

logger = logging.getLogger(__name__)

# =====================================================================
# Series Processor Class
# =====================================================================

class SeriesProcessor:
    """Processes a series of XRD images to remove cosmic ray artifacts.
    
    This class provides functionality for processing series of XRD images to
    remove cosmic ray artifacts. It supports both single image processing and
    batch processing of entire series.
    
    Attributes:
        first_filename (Path): Path to the first image or directory containing images
        user_mask (np.ndarray): User-defined mask for valid pixels, where True means can be modified
        use_fabio (bool): Whether to use fabio.open_series instead of manual loading
        img_series (ImageSeries): Image series object for managing the collection of images
        img_num (int): Total number of images in the series
        img_shape (Tuple[int, int]): Dimensions of each image (height, width)
        img_dtype (np.dtype): Data type of the image arrays (e.g., np.float32, np.int32)
        
        # Processing Parameters
        th_donut (int): Threshold for detecting donut-shaped cosmic ray artifacts
        th_mask (float): Threshold for creating the ring artifact mask
        th_streak (int): Threshold for detecting streak-shaped cosmic ray artifacts
        win_streak (int): Size of the sliding window for streak detection
        exp_donut (int): Number of pixels to expand donut masks
        exp_streak (int): Number of pixels to expand streak masks
        
        # Processing Results
        img_avg (np.ndarray): Mean intensity across all images
        img_binary_avg (np.ndarray): Mean of thresholded binary images
        ring_mask (np.ndarray): Boolean mask protecting ring features
        combined_mask (np.ndarray): Logical AND of user_mask and ring_mask
        img_clean_avg (np.ndarray): Mean intensity of cleaned images
        sub_donut_avg (np.ndarray): Mean of removed donut artifacts
        sub_streak_avg (np.ndarray): Mean of removed streak artifacts
    """
    
    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self,
                first_filename: str,
                th_donut: int,
                th_mask: float,
                th_streak: int,
                win_streak: int,
                exp_donut: int,
                exp_streak: int,
                user_mask: Optional[np.ndarray] = None,
                use_fabio: bool = False,
                calc_std: bool = False
                ) -> None:
        """Initialize the processor.
        
        Args:
            first_filename: Path to the first image or directory containing images
            user_mask: User-defined mask for valid pixels (default: True)
            use_fabio: Whether to use fabio.open_series instead of manual loading
            th_donut: Threshold for donut detection
            th_mask: Threshold for mask creation
            th_streak: Threshold for streak detection
            win_streak: Window size for streak detection
            exp_donut: Expansion of donut mask
            exp_streak: Expansion of streak mask
        """
        try:
            self.first_filename = Path(first_filename).resolve()
            self._load_images(self.first_filename, use_fabio)
            self.user_mask = user_mask
            self.calc_std = calc_std

            # Processing parameters
            self.th_donut = th_donut
            self.th_mask = th_mask
            self.th_streak = th_streak
            self.win_streak = win_streak
            self.exp_donut = exp_donut
            self.exp_streak = exp_streak

            # Intermediate results
            self.img_binary_avg = None
            self.ring_mask = None
            self.img_intermediate_num = None
            self.img_clean_num = None

            # Output results
            self.img_avg = None
            self.img_intermediate_avg = None
            self.img_clean_avg = None
            self.combined_mask = None
            self.sub_donut_avg = None
            self.sub_streak_avg = None

            if self.calc_std:
                self.img_std = None
                self.img_std_intermediate = None
                self.img_std_clean = None
            
        except Exception as e:
            logger.error(f"Failed to initialize SeriesProcessor: {str(e)}")
            raise

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _load_images(self, first_filename: Path, use_fabio: bool = False) -> None:
        """Load images from a file.
        
        Args:
            first_filename: Path to the first image or directory containing images
            use_fabio: Whether to use fabio.open_series instead of manual loading
        """
        try:
            logger.info(f"Loading {first_filename} ...")
            
            # Create image series using the new ImageSeries class
            self.img_series = ImageSeries.create(first_filename, use_fabio)
                
            logger.info(f"Loaded {self.img_series.nframes} images")
            self.img_num = self.img_series.nframes
            
            # Get shape and dtype from first frame
            first_frame = self.img_series.get_frame(0)
            self.img_shape = first_frame.shape
            self.img_dtype = first_frame.dtype
        except Exception as e:
            logger.error(f"Failed to load images from {first_filename}: {str(e)}")
            raise

    def _get_img(self, idx: int) -> np.ndarray:
        """Get a single image from the series.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            np.ndarray: Processed image as numpy array
        """
        try:
            img = self.img_series.get_frame(idx)
            img = img.astype(np.int32)
            img = np.nan_to_num(img, nan=-1)
            img = np.clip(img, 0, None)
            return img
        except Exception as e:
            logger.error(f"Failed to get image at index {idx}: {str(e)}")
            raise
    
    def _avg_img(self) -> None:
        """Calculate average image and binary average.
        
        This method calculates the average of all images and the average of
        binary images (where each pixel is 1 if the original pixel is > 0).
        """
        try:
            img_sum = np.zeros(self.img_shape, dtype=np.float64)
            img_binary_sum = np.zeros(self.img_shape, dtype=np.float64)
            logger.info('Averaging images ...')
            
            for i in tqdm(range(self.img_num), desc='Averaging images'):
                img = self._get_img(i)
                img_sum += img
                img_binary = img > 0
                img_binary_sum += img_binary
            
            self.img_avg = img_sum / self.img_num
            self.img_binary_avg = img_binary_sum / self.img_num
            logger.debug("Average image calculated")
        except Exception as e:
            logger.error(f"Failed to calculate average image: {str(e)}")
            raise

    def _mask_img(self) -> None:
        """Create mask for ring features.
        
        This method creates a mask to protect ring features by identifying pixels that appear
        consistently across the image series (using the binary average) and combining this
        with any user-specified mask regions.
        """
        try:
            if self.img_binary_avg is None:
                raise ValueError("Binary average image not calculated")
            
            self.ring_mask = self.img_binary_avg < self.th_mask
            logger.debug("Number of pixels masked as ring features: %d", np.sum(self.ring_mask))

            if self.user_mask is not None:
                self.combined_mask = self.ring_mask & self.user_mask
            else:
                self.combined_mask = self.ring_mask

            logger.debug("Combined mask created")
        except Exception as e:
            logger.error(f"Failed to create mask: {str(e)}")
            raise

    def _avg_clean_img(self) -> None:
        """Process all images to remove cosmic ray artifacts.
        
        This method processes all images in the series to remove cosmic ray
        artifacts and calculates various averages and differences.
        """
        try:
            if self.combined_mask is None:
                raise ValueError("Combined mask not calculated")
            
            img_intermediate_sum = np.zeros(self.img_shape, dtype=np.float64)
            img_clean_sum = np.zeros(self.img_shape, dtype=np.float64)
            sub_donut_sum = np.zeros(self.img_shape, dtype=np.float64)
            sub_streak_sum = np.zeros(self.img_shape, dtype=np.float64)
            self.img_intermediate_num = np.ones(self.img_shape, dtype=np.int32) * self.img_num
            self.img_clean_num = np.ones(self.img_shape, dtype=np.int32) * self.img_num
            
            logger.info('Cleaning images ...')
            for i in tqdm(range(self.img_num), desc='Cleaning images'):
                img = self._get_img(i)
                processor = ImageProcessor(img,
                                        self.combined_mask,
                                        self.th_donut,
                                        self.th_streak,
                                        self.win_streak,
                                        self.exp_donut,
                                        self.exp_streak)
                processor.clean_img()
                
                img_intermediate_sum += processor.img_intermediate
                img_clean_sum += processor.img_clean
                sub_donut_sum += processor.sub_donut
                sub_streak_sum += processor.sub_streak
                self.img_intermediate_num -= processor.mask_donut
                self.img_clean_num -= processor.mod_mask
            
            self.img_intermediate_avg = np.divide(img_intermediate_sum, self.img_intermediate_num, out=np.zeros_like(img_intermediate_sum), where=self.img_intermediate_num != 0)
            self.img_clean_avg = np.divide(img_clean_sum, self.img_clean_num, out=np.zeros_like(img_clean_sum), where=self.img_clean_num != 0)
            self.sub_donut_avg = np.divide(sub_donut_sum, self.img_num, out=np.zeros_like(sub_donut_sum), where=self.img_num != 0)
            self.sub_streak_avg = np.divide(sub_streak_sum, self.img_num, out=np.zeros_like(sub_streak_sum), where=self.img_num != 0)
            logger.debug("Cleaned average image calculated")
        except Exception as e:
            logger.error(f"Failed to clean images: {str(e)}")
            raise

    def _std_avg_img(self) -> None:
        """Calculate standard deviation of images during averaging.
        
        This method calculates the standard deviation of all images in the series
        while computing the average, avoiding loading the entire series twice.
        """
        try:
            if self.img_avg is None:
                raise ValueError("Average image not calculated")
            
            img_sum_sq = np.zeros(self.img_shape, dtype=np.float64)
            
            for i in tqdm(range(self.img_num), desc='Calculating standard deviation of images'):
                img = self._get_img(i)
                img_sum_sq += (img - self.img_avg) ** 2
                
            self.img_std = np.sqrt(img_sum_sq / self.img_num)
            logger.debug("Standard deviation of images calculated")
        except Exception as e:
            logger.error(f"Failed to calculate standard deviation: {str(e)}")
            raise

    def _std_avg_clean_img(self) -> None:
        """Calculate standard deviation of cleaned images during averaging.
        
        This method calculates the standard deviation of all cleaned images in the series
        while computing the average, avoiding loading the entire series twice.
        """
        try:
            if self.img_clean_avg is None:
                raise ValueError("Cleaned average image not calculated")
            
            img_intermediate_sum_sq = np.zeros(self.img_shape, dtype=np.float64)
            img_clean_sum_sq = np.zeros(self.img_shape, dtype=np.float64)
            
            for i in tqdm(range(self.img_num), desc='Calculating standard deviation of cleaned images'):
                img = self._get_img(i)
                processor = ImageProcessor(img,
                                           self.combined_mask,
                                           self.th_donut,
                                           self.th_streak,
                                           self.win_streak,
                                           self.exp_donut,
                                           self.exp_streak)
                processor.clean_img()

                img_intermediate_sum_sq += (processor.img_intermediate - self.img_intermediate_avg) ** 2
                img_clean_sum_sq += (processor.img_clean - self.img_clean_avg) ** 2
            
            self.img_std_intermediate = np.sqrt(np.divide(img_intermediate_sum_sq, self.img_intermediate_num, out=np.zeros_like(img_intermediate_sum_sq), where=self.img_intermediate_num != 0))
            self.img_std_clean = np.sqrt(np.divide(img_clean_sum_sq, self.img_clean_num, out=np.zeros_like(img_clean_sum_sq), where=self.img_clean_num != 0))
            logger.debug("Standard deviation of cleaned images calculated")
        except Exception as e:
            logger.error(f"Failed to calculate standard deviation of cleaned images: {str(e)}")
            raise
        
    # =====================================================================
    # Public Methods
    # =====================================================================

    def process_single(self, idx: int) -> ImageProcessor:
        """Process a single image. For debugging purposes.
        
        Args:
            idx: Index of the image to clean
            
        Returns:
            ImageProcessor: Processor instance containing the cleaned image
        """
        try:
            if self.img_avg is None:
                self._avg_img()
            if self.combined_mask is None:
                self._mask_img()

            logger.info(f"Cleaning image {idx} ...")
            img = self._get_img(idx)
            processor = ImageProcessor(img, self.combined_mask)
            processor.clean_img()
            logger.debug("Single image cleaned")
            return processor
        except Exception as e:
            logger.error(f"Failed to clean single image at index {idx}: {str(e)}")
            raise

    def process(self) -> None:
        """Main processing pipeline for cosmic ray removal.
        
        This method orchestrates the complete processing workflow:
        1. Calculate average of all images
        2. Generate combined mask for valid pixels
        3. Process images to remove cosmic rays and calculate averages
        4. Compute standard deviation across the series
        """
        try:
            self._avg_img()
            if self.calc_std:
                self._std_avg_img()
            self._mask_img()
            self._avg_clean_img()
            if self.calc_std:
                self._std_avg_clean_img()
        except Exception as e:
            logger.error(f"Failed to process image series: {str(e)}")
            raise

    def save_results(self, output_dir: str, prefix: str = '') -> None:
        """Save processing results as TIFF files.
        
        This method saves the results of processing the entire series, including
        averaged images, cleaned averages, differences, masks, and subtracted
        components.
        
        Args:
            output_dir: Directory to save the results
        """
        try:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save direct averaged images
            fabio.tifimage.tifimage(data=self.img_avg).write(
                output_dir / f'{prefix}_avg.tif'
            )

            # Save intermediate averaged images
            fabio.tifimage.tifimage(data=self.img_intermediate_avg).write(
                output_dir / f'{prefix}_intermediate.tif'
            )

            # Save cleaned averaged images
            fabio.tifimage.tifimage(data=self.img_clean_avg).write(
                output_dir / f'{prefix}_clean.tif'
            )
            
            # Save masks
            fabio.tifimage.tifimage(data=self.combined_mask.astype(np.int32)).write(
                output_dir / f'{prefix}_mask.tif'
            )
            
            # Save subtracted components
            fabio.tifimage.tifimage(data=self.sub_donut_avg).write(
                output_dir / f'{prefix}_donut.tif'
            )

            fabio.tifimage.tifimage(data=self.sub_streak_avg).write(
                output_dir / f'{prefix}_streak.tif'
            ) 

            # Save standard deviation of images
            fabio.tifimage.tifimage(data=self.img_std).write(
                output_dir / f'{prefix}_std.tif'
            )

            # Save standard deviation of intermediate images
            fabio.tifimage.tifimage(data=self.img_std_intermediate).write(
                output_dir / f'{prefix}_std_intermediate.tif'
            )

            # Save standard deviation of cleaned images 
            fabio.tifimage.tifimage(data=self.img_std_clean).write(
                output_dir / f'{prefix}_std_clean.tif'
            )

            metadata = {
                'data': {
                    'path': os.path.dirname(self.first_filename),
                    'img_num': self.img_num,
                    'img_shape': self.img_shape,
                    'img_dtype': str(self.img_dtype)
                },
                'parameters': {
                    'th_donut': self.th_donut,
                    'th_mask': self.th_mask,
                    'th_streak': self.th_streak,
                    'win_streak': self.win_streak,
                    'exp_donut': self.exp_donut,
                    'exp_streak': self.exp_streak
                }
            }
            # Save parameters
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Failed to save results to {output_dir}: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources when the processor is deleted."""
        try:
            if hasattr(self, 'img_series'):
                self.img_series.cleanup()
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {str(e)}")
            raise
