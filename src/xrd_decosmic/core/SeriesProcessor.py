"""
Processor for handling series of XRD images.

This module provides functionality for processing series of XRD images to remove
cosmic ray artifacts. It supports both single image processing and batch processing
of entire series.
"""
import os
import fabio
import numpy as np
from typing import Callable, Optional

from .ImageProcessor import ImageProcessor
from .ImageSeries import ImageSeries

# =====================================================================
# Series Processor Class
# =====================================================================

class SeriesProcessor:
    """Processes a series of XRD images to remove cosmic ray artifacts.
    
    This class provides functionality for processing series of XRD images to
    remove cosmic ray artifacts. It supports both single image processing and
    batch processing of entire series.
    
    Attributes:
        first_filename (str): Path to the first image or directory containing images
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
                user_mask: np.ndarray,
                use_fabio: bool = False
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
        self.first_filename = first_filename
        self.load_images(first_filename, use_fabio)
        self.user_mask = user_mask

        # Processing parameters
        self.th_donut = th_donut
        self.th_mask = th_mask
        self.th_streak = th_streak
        self.win_streak = win_streak
        self.exp_donut = exp_donut
        self.exp_streak = exp_streak
        
        # Processing results
        self.img_avg = None
        self.img_binary_avg = None
        self.ring_mask = None
        self.combined_mask = None
        self.img_intermediate_avg = None
        self.img_clean_avg = None
        self.sub_donut_avg = None
        self.sub_streak_avg = None

    def __del__(self) -> None:
        """Clean up resources when the processor is deleted."""
        if hasattr(self, 'img_series'):
            self.img_series.cleanup()

    # =====================================================================
    # Public Methods
    # =====================================================================

    def load_images(self, first_filename: str, use_fabio: bool = False) -> None:
        """Load images from a file.
        
        Args:
            first_filename: Path to the first image or directory containing images
            use_fabio: Whether to use fabio.open_series instead of manual loading
        """
        print(f"Loading {first_filename} ...")
        
        # Create image series using the new ImageSeries class
        self.img_series = ImageSeries.create(first_filename, use_fabio)
            
        print(f"Loaded {self.img_series.nframes} images")
        self.img_num = self.img_series.nframes
        
        # Get shape and dtype from first frame
        first_frame = self.img_series.get_frame(0)
        self.img_shape = first_frame.shape
        self.img_dtype = first_frame.dtype

    def get_img(self, idx: int) -> np.ndarray:
        """Get a single image from the series.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            np.ndarray: Processed image as numpy array
        """
        img = self.img_series.get_frame(idx)
        img = img.astype(np.int32)
        img = np.nan_to_num(img, nan=-1)
        img = np.clip(img, 0, None)
        return img
    
    def avg_img(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Calculate average image and binary average.
        
        This method calculates the average of all images and the average of
        binary images (where each pixel is 1 if the original pixel is > 0).
        
        Args:
            progress_callback: Optional callback function to report progress
        """
        img_sum = np.zeros(self.img_shape, dtype=np.float64)
        img_binary_sum = np.zeros(self.img_shape, dtype=np.float64)
        print('Averaging images ...')
        
        last_progress = -1
        for i in range(self.img_num):
            img = self.get_img(i)
            img_sum += img
            img_binary = img > 0
            img_binary_sum += img_binary
            if progress_callback:
                current_progress = (i * 100) // self.img_num
                if current_progress // 10 > last_progress // 10:
                    progress_callback(current_progress)
                    last_progress = current_progress
        
        # Show 100% at the end
        if progress_callback:
            progress_callback(100)
        
        self.img_avg = img_sum / self.img_num
        self.img_binary_avg = img_binary_sum / self.img_num

    def mask_img(self) -> None:
        """Create mask for ring artifacts.
        
        This method creates a mask for ring artifacts based on the binary
        average image and combines it with the user mask.
        """
        self.ring_mask = self.img_binary_avg < self.th_mask
        self.combined_mask = self.ring_mask & self.user_mask

    def single_clean_img(self, idx: int, progress_callback: Optional[Callable[[int], None]] = None) -> ImageProcessor:
        """Clean a single image.
        
        Args:
            idx: Index of the image to clean
            progress_callback: Optional callback function to report progress
            
        Returns:
            ImageProcessor: Processor instance containing the cleaned image
        """
        if self.img_avg is None:
            self.avg_img(progress_callback)
        if self.combined_mask is None:
            self.mask_img()

        print(f"Cleaning image {idx} ...")
        img = self.get_img(idx)
        processor = ImageProcessor(img, self.combined_mask)
        processor.load_params(
            self.th_donut, self.th_streak, self.win_streak,
            self.exp_donut, self.exp_streak)
        processor.clean_img()
        return processor

    def avg_clean_img(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Process all images to remove cosmic ray artifacts.
        
        This method processes all images in the series to remove cosmic ray
        artifacts and calculates various averages and differences.
        
        Args:
            progress_callback: Optional callback function to report progress
        """
        self.avg_img(progress_callback)
        self.mask_img()
        
        img_clean_sum = np.zeros(self.img_shape, dtype=np.float64)
        img_clean_num = np.ones(self.img_shape, dtype=np.int32) * self.img_num
        sub_donut_sum = np.zeros(self.img_shape, dtype=np.float64)
        sub_streak_sum = np.zeros(self.img_shape, dtype=np.float64)
        
        print('Cleaning images ...')
        last_progress = -1
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
                current_progress = (i * 100) // self.img_num
                if current_progress // 10 > last_progress // 10:
                    progress_callback(current_progress)
                    last_progress = current_progress
        
        # Show 100% at the end
        if progress_callback:
            progress_callback(100)
        
        self.img_clean_avg = img_clean_sum / img_clean_num
        self.sub_donut_avg = sub_donut_sum / self.img_num
        self.sub_streak_avg = sub_streak_sum / self.img_num
        self.img_diff_avg = self.img_avg - self.img_clean_avg

    # =====================================================================
    # File Operations
    # =====================================================================

    def save_single_result(self, idx: int, output_dir: str) -> None:
        """Save single result.
        
        This method saves the results of processing a single image, including
        the raw image, intermediate image, cleaned image, modification mask,
        and subtracted components.
        
        Args:
            idx: Index of the processed image
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        processor = self.single_clean_img(idx, lambda p: print(f'Progress: {p}%'))

        # Save raw image
        fabio.tifimage.tifimage(data=processor.img).write(
            os.path.join(output_dir, f'{idx:04d}.tif')
        )

        # Save intermediate image
        fabio.tifimage.tifimage(data=processor.img_intermediate).write(
            os.path.join(output_dir, f'{idx:04d}_intermediate.tif')
        )

        # Save cleaned image
        fabio.tifimage.tifimage(data=processor.img_clean).write(
            os.path.join(output_dir, f'{idx:04d}_clean.tif')
        )

        # Save mask
        fabio.tifimage.tifimage(data=processor.mod_mask.astype(np.int32)).write(
            os.path.join(output_dir, f'{idx:04d}_mod_mask.tif')
        )

        # Save subtracted components
        fabio.tifimage.tifimage(data=processor.sub_donut).write(
            os.path.join(output_dir, f'{idx:04d}_donut.tif')
        )
        fabio.tifimage.tifimage(data=processor.sub_streak).write(
            os.path.join(output_dir, f'{idx:04d}_streak.tif')
        )

    def save_results(self, output_dir: str) -> None:
        """Save processing results as TIFF files.
        
        This method saves the results of processing the entire series, including
        averaged images, cleaned averages, differences, masks, and subtracted
        components.
        
        Args:
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save direct averaged images
        fabio.tifimage.tifimage(data=self.img_avg).write(
            os.path.join(output_dir, 'avg.tif')
        )

        # Save intermediate averaged images
        fabio.tifimage.tifimage(data=self.img_intermediate_avg).write(
            os.path.join(output_dir, 'intermediate.tif')
        )

        # Save cleaned averaged images
        fabio.tifimage.tifimage(data=self.img_clean_avg).write(
            os.path.join(output_dir, 'clean.tif')
        )
        
        # Save masks
        fabio.tifimage.tifimage(data=self.combined_mask.astype(np.int32)).write(
            os.path.join(output_dir, 'mask.tif')
        )
        
        # Save subtracted components
        fabio.tifimage.tifimage(data=self.sub_donut_avg).write(
            os.path.join(output_dir, 'donut.tif')
        )
        fabio.tifimage.tifimage(data=self.sub_streak_avg).write(
            os.path.join(output_dir, 'streak.tif')
        ) 