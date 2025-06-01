"""
Comment: Series image processing
Dataclass: SeriesConfig, SeriesResult
Class: SeriesProcessor
"""
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
import numpy as np
from pathlib import Path

import tifffile
from tqdm import tqdm

from .single_processor import SingleProcessor, SingleConfig
from .image_series import ImageSeries

logger = logging.getLogger(__name__)

# =====================================================================
# Config and Result Dataclasses
# =====================================================================

@dataclass
class SeriesConfig(SingleConfig):
    """Configuration for SeriesProcessor.
    
    Attributes:
        th_mask (float): Threshold for creating mask_protect
    """
    th_mask: float

@dataclass
class SeriesResult:
    """Results of the series processing.
    
    Attributes:
        img_avg_direct (np.ndarray): Direct average
        img_avg_binary (np.ndarray): Direct average of binary images
        img_avg_half_clean (np.ndarray): Half-cleaned average
        img_avg_clean (np.ndarray): Cleaned average
        sub_avg_donut (np.ndarray): Donut subtracted average
        sub_avg_streak (np.ndarray): Streak subtracted average
        mask_protect (np.ndarray): Boolean mask to protect ring features, where True means can be modified
        mask_modifiable (np.ndarray): Logical AND of mask_user and mask_protect, where True means can be modified
        img_err_direct (np.ndarray): Error of direct average
        img_err_half_clean (np.ndarray): Error of half-cleaned average
        img_err_clean (np.ndarray): Error of cleaned average
    """
    img_avg_direct: np.ndarray | None = None
    img_avg_binary: np.ndarray | None = None
    mask_protect: np.ndarray | None = None
    mask_modifiable: np.ndarray | None = None
    img_avg_clean: np.ndarray | None = None
    img_avg_half_clean: np.ndarray | None = None
    sub_avg_donut: np.ndarray | None = None
    sub_avg_streak: np.ndarray | None = None
    img_err_direct: np.ndarray | None = None
    img_err_half_clean: np.ndarray | None = None
    img_err_clean: np.ndarray | None = None

    def save(self, output_dir: str, prefix: str = '') -> None:
        """Save the results to a file.
        
        Args:
            output_path: Path to the output file
        """
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in self.__dict__.items():
            if value is not None:
                tifffile.imwrite(
                    output_path / f'{prefix}_{key}.tif',
                    value
                )

    def load(self, input_dir: str, prefix: str = '') -> None:
        """Load the results from a file.
        
        Args:
            input_path: Path to the input file
        """
        input_path = Path(input_dir).resolve()
        for key in self.__dict__:
            file_path = input_path / f'{prefix}_{key}.tif'
            if file_path.exists():
                setattr(self, key, tifffile.imread(file_path))
            else:
                raise FileNotFoundError(f"File {file_path} does not exist")

# =====================================================================
# Series Processor Class
# =====================================================================

class SeriesProcessor:
    """Processes a series of images to remove high energy background.
    
    This class implements different averaging algorithms and can calculate errors of the averages (standard deviation).
    
    Attributes:
        config (SeriesConfig): Configuration for the series processor
        result (SeriesResult): Results of the series processing
        img_series (ImageSeries): Image series object for managing the collection of images
        nframes (int): Number of frames in the series
        shape (Tuple[int, int]): Shape of the images
        dtype (np.dtype): Data type of the images
    """
    
    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self,
                first_filename: str,
                series_config: SeriesConfig,
                mask_modifiable: np.ndarray | None = None,
                use_fabio: bool = False
                ) -> None:
        """Initialize the processor.
        
        Args:
            first_filename: Path to the first image or directory containing images
            series_config: Configuration for the series processor
            mask_modifiable: User-defined mask for modifiable pixels, where True means can be modified
            use_fabio: Whether to use fabio.open_series instead of manual loading
        """
        try:
            self.series_result = SeriesResult(
                mask_modifiable=mask_modifiable
            )
            self.series_config = series_config
            self.first_filename = str(Path(first_filename).resolve())
            if not os.path.isfile(self.first_filename):
                raise FileNotFoundError(f"File {self.first_filename} not found")
            self._load_images(self.first_filename, use_fabio=use_fabio)

            logger.debug(f"SeriesProcessor initialized.")
            logger.debug(f"Configuration: {self.series_config}")
        except Exception as e:
            logger.error(f"Failed to initialize SeriesProcessor: {str(e)}")
            raise

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _load_images(self, first_filename: str, use_fabio: bool = False) -> None:
        """Load images from a file.
        
        Args:
            first_filename: Path to the first image of the series
            use_fabio: Whether to use fabio.open_series instead of manual loading
        """
        try:
            logger.info(f"Loading images from {first_filename} ...")
            
            # Create image series using the new ImageSeries class
            self.img_series = ImageSeries.create(first_filename, use_fabio)
            
            # Get number of frames
            self.nframes = self.img_series.nframes
            
            # Get shape and dtype from first frame
            first_frame = self.img_series.get_frame(0)
            self.shape = first_frame.shape
            self.dtype = first_frame.dtype

            logger.info(f"Loaded {self.nframes} images.")
            logger.info(f"Image shape: {self.shape}")
            logger.info(f"Image dtype: {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load images from {first_filename}: {str(e)}")
            raise

    def _get_img(self, idx: int) -> np.ndarray:
        """Get a single image from the series.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            np.ndarray: Preprocessed image as numpy array
        """
        try:
            img = self.img_series.get_frame(idx)
            img[img>10000] = 0 # Big values are set to 0
            img = np.nan_to_num(img, nan=0) # NaN values are set to 0
            img = np.clip(img, 0, None) # Negative values are set to 0
            return img
        except Exception as e:
            logger.error(f"Failed to get image at index {idx}: {str(e)}")
            raise
    
    def _avg_direct(self) -> None:
        """Calculate direct average
        
        This method calculates the direct average of all images and their binarization.
        """
        try:
            img_sum_direct = np.zeros(self.shape, dtype=np.float64)
            img_sum_binary = np.zeros(self.shape, dtype=np.float64)
            logger.info('Direct averaging images ...')
            
            for i in tqdm(range(self.nframes), desc='Direct-averaging images'):
                img = self._get_img(i)
                img_sum_direct += img

                img_binary = (img > 0).astype(self.dtype)
                img_sum_binary += img_binary
            
            self.series_result.img_avg_direct = img_sum_direct / self.nframes
            self.series_result.img_avg_binary = img_sum_binary / self.nframes
            logger.debug("Direct-average finished")
        except Exception as e:
            logger.error(f"Direct-average failed: {str(e)}")
            raise

    def _mask(self) -> None:
        """Create protect mask for ring features.
        
        This method creates a mask to protect ring features by identifying pixels that appear consistently across the image series (using the binary average) and combining this with any user-specified mask regions.
        """
        try:
            if self.series_result.img_avg_binary is None:
                raise ValueError("Binary average image not calculated")
            
            self.series_result.mask_protect = self.series_result.img_avg_binary <= self.series_config.th_mask
            logger.debug(f"Number of pixels protected as ring features: {np.sum(1 - self.series_result.mask_protect)}")

            if self.series_result.mask_modifiable is not None:
                self.series_result.mask_modifiable = self.series_result.mask_protect & self.series_result.mask_modifiable
            else:
                self.series_result.mask_modifiable = self.series_result.mask_protect

            logger.debug("Combined mask created")
        except Exception as e:
            logger.error(f"Failed to create mask: {str(e)}")
            raise

    def _avg_clean(self) -> None:
        """Process all images to remove high energy background.
        
        This method processes all images in the series to remove high energy background
        and calculates various averages and differences.
        """
        try:
            if self.series_result.mask_modifiable is None:
                raise ValueError("Modifiable mask not calculated")
            
            img_sum_half_clean = np.zeros(self.shape, dtype=np.float64)
            img_sum_clean = np.zeros(self.shape, dtype=np.float64)
            sub_sum_donut = np.zeros(self.shape, dtype=np.float64)
            sub_sum_streak = np.zeros(self.shape, dtype=np.float64)
            num_half_clean = np.ones(self.shape, dtype=np.int32) * self.nframes
            num_clean = np.ones(self.shape, dtype=np.int32) * self.nframes
            
            logger.info('Cleaning images ...')
            for i in tqdm(range(self.nframes), desc='Cleaning images'):
                img = self._get_img(i)
                processor = SingleProcessor(
                    img,
                    self.series_config,
                    self.series_result.mask_modifiable
                )
                single_result = processor.clean_img()
                
                img_sum_half_clean += single_result.img_half_clean
                img_sum_clean += single_result.img_clean
                sub_sum_donut += single_result.sub_donut
                sub_sum_streak += single_result.sub_streak
                num_half_clean -= single_result.mask_donut
                num_clean -= single_result.mask_modified
            
            self.series_result.img_avg_half_clean = np.divide(img_sum_half_clean, num_half_clean, out=np.zeros_like(img_sum_half_clean), where=num_half_clean != 0)
            self.series_result.img_avg_clean = np.divide(img_sum_clean, num_clean, out=np.zeros_like(img_sum_clean), where=num_clean != 0)
            self.series_result.sub_avg_donut = sub_sum_donut / self.nframes
            self.series_result.sub_avg_streak = sub_sum_streak / self.nframes
            logger.debug("Cleaning finished")
        except Exception as e:
            logger.error(f"Cleaning failed: {str(e)}")
            raise

    def _err_direct(self) -> None:
        """Calculate error of direct average.
        
        This method calculates the error of direct average.
        """
        try:
            if self.series_result.img_avg_direct is None:
                raise ValueError("Direct average not calculated")
            
            img_sum_sq = np.zeros(self.shape, dtype=np.float64)
            
            for i in tqdm(range(self.nframes), desc='Calculating error of direct average'):
                img = self._get_img(i)
                img_sum_sq += (img - self.series_result.img_avg_direct) ** 2
                
            self.series_result.img_err_direct = np.sqrt(img_sum_sq / self.nframes)
            logger.debug("Error of direct average calculated")
        except Exception as e:
            logger.error(f"Failed to calculate error of direct average: {str(e)}")
            raise

    def _err_clean(self) -> None:
        """Calculate error of cleaned average.
        
        This method calculates the error of cleaned average.
        """
        try:
            if self.series_result.img_avg_clean is None:
                raise ValueError("Cleaned average not calculated")
            
            img_sum_sq_half_clean = np.zeros(self.shape, dtype=np.float64)
            img_sum_sq_clean = np.zeros(self.shape, dtype=np.float64)
            
            for i in tqdm(range(self.nframes), desc='Calculating error of cleaned average'):
                img = self._get_img(i)
                processor = SingleProcessor(
                    img,
                    self.series_config,
                    self.series_result.mask_modifiable
                )
                single_result = processor.clean_img()

                if single_result.img_half_clean is None:
                    raise ValueError("Half-cleaned image not calculated")
                if single_result.img_clean is None:
                    raise ValueError("Cleaned image not calculated")

                img_sum_sq_half_clean += (single_result.img_half_clean - self.series_result.img_avg_half_clean) ** 2
                img_sum_sq_clean += (single_result.img_clean - self.series_result.img_avg_clean) ** 2

            self.series_result.img_err_half_clean = np.sqrt(img_sum_sq_half_clean / self.nframes)
            self.series_result.img_err_clean = np.sqrt(img_sum_sq_clean / self.nframes)
            logger.debug("Error of cleaned average calculated")
        except Exception as e:
            logger.error(f"Failed to calculate error of cleaned average: {str(e)}")
            raise
        
    # =====================================================================
    # Public Methods
    # =====================================================================

    def process_series(self) -> SeriesResult:
        """Main processing pipeline for decosmic.
        
        This method orchestrates the complete processing workflow:
        1. Calculate direct average of all images
        2. Calculate error of direct average
        3. Generate combined mask for valid pixels
        4. Process images to remove high energy background and calculate averages
        5. Calculate error of cleaned average
        """
        try:
            self._avg_direct()
            self._err_direct()
            self._mask()
            self._avg_clean()
            self._err_clean()
            logger.info('Processing finished')

            return deepcopy(self.series_result)
        except Exception as e:
            logger.error(f"Failed to process image series: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources when the processor is deleted."""
        try:
            if hasattr(self, 'img_series'):
                self.img_series.cleanup()
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {str(e)}")
            raise
