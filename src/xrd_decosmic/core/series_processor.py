"""Series image processing module with SeriesConfig, SeriesResult dataclasses and SeriesProcessor class."""
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
    """Configuration parameters for SeriesProcessor extending SingleConfig."""
    th_mask: float

@dataclass
class SeriesResult:
    """Results container for series processing with averages, variances and masks."""
    avg_direct: np.ndarray | None = None
    avg_binary: np.ndarray | None = None
    mask_protect: np.ndarray | None = None
    mask_modifiable: np.ndarray | None = None
    avg_clean: np.ndarray | None = None
    avg_half_clean: np.ndarray | None = None
    avg_donut: np.ndarray | None = None
    avg_streak: np.ndarray | None = None
    var_direct: np.ndarray | None = None
    var_half_clean: np.ndarray | None = None
    var_clean: np.ndarray | None = None

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
# Series Processor Class
# =====================================================================

class SeriesProcessor:
    """Processes image series to remove high energy background with averaging algorithms and variance calculation."""
    
    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self,
                first_filename: str,
                series_config: SeriesConfig,
                mask_modifiable: np.ndarray | None = None,
                use_fabio: bool = False
                ) -> None:
        """Initialize the processor with first filename, configuration, optional mask and fabio usage flag."""
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
        """Load images from file series using the ImageSeries factory."""
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
        """Get a single preprocessed image from the series with outlier value handling."""
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
        """Calculate direct average of all images and their binary representations."""
        try:
            sum_direct = np.zeros(self.shape, dtype=np.float64)
            sum_binary = np.zeros(self.shape, dtype=np.float64)
            logger.info('Direct averaging images ...')
            
            for i in tqdm(range(self.nframes), desc='Direct-averaging images'):
                img = self._get_img(i)
                sum_direct += img

                img_binary = (img > 0).astype(self.dtype)
                sum_binary += img_binary
            
            self.series_result.avg_direct = sum_direct / self.nframes
            self.series_result.avg_binary = sum_binary / self.nframes
            logger.debug("Direct-average finished")
        except Exception as e:
            logger.error(f"Direct-average failed: {str(e)}")
            raise

    def _mask(self) -> None:
        """Create protection mask for ring features based on binary average threshold."""
        try:
            if self.series_result.avg_binary is None:
                raise ValueError("Binary average image not calculated")
            
            self.series_result.mask_protect = self.series_result.avg_binary <= self.series_config.th_mask
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
        """Process all images to remove high energy background and calculate cleaned averages."""
        try:
            if self.series_result.mask_modifiable is None:
                raise ValueError("Modifiable mask not calculated")
            
            sum_half_clean = np.zeros(self.shape, dtype=np.float64)
            sum_clean = np.zeros(self.shape, dtype=np.float64)
            sum_donut = np.zeros(self.shape, dtype=np.float64)
            sum_streak = np.zeros(self.shape, dtype=np.float64)
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
                
                if single_result.img_half_clean is not None:
                    sum_half_clean += single_result.img_half_clean
                if single_result.img_clean is not None:
                    sum_clean += single_result.img_clean
                if single_result.sub_donut is not None:
                    sum_donut += single_result.sub_donut
                if single_result.sub_streak is not None:
                    sum_streak += single_result.sub_streak
                if single_result.mask_donut is not None:
                    num_half_clean -= single_result.mask_donut.astype(np.int32)
                if single_result.mask_combined is not None:
                    num_clean -= single_result.mask_combined.astype(np.int32)
            
            self.series_result.avg_half_clean = np.divide(sum_half_clean, num_half_clean, out=np.zeros_like(sum_half_clean), where=num_half_clean != 0)
            self.series_result.avg_clean = np.divide(sum_clean, num_clean, out=np.zeros_like(sum_clean), where=num_clean != 0)
            self.series_result.avg_donut = sum_donut / self.nframes
            self.series_result.avg_streak = sum_streak / self.nframes
            logger.debug("Clean-average finished")
        except Exception as e:
            logger.error(f"Clean-average failed: {str(e)}")
            raise

    def _var_direct(self) -> None:
        """Calculate variance of direct average across all images."""
        try:
            if self.series_result.avg_direct is None:
                raise ValueError("Direct average not calculated")
            
            sum_variance = np.zeros(self.shape, dtype=np.float64)
            logger.info('Calculating direct variance ...')
            
            for i in tqdm(range(self.nframes), desc='Calculating direct variance'):
                img = self._get_img(i)
                diff = img - self.series_result.avg_direct
                sum_variance += diff ** 2
            
            self.series_result.var_direct = sum_variance / self.nframes
            logger.debug("Direct-variance calculated")
        except Exception as e:
            logger.error(f"Direct-variance calculation failed: {str(e)}")
            raise

    def _var_clean(self) -> None:
        """Calculate variance of cleaned average across all processed images."""
        try:
            if self.series_result.avg_clean is None or self.series_result.mask_modifiable is None:
                raise ValueError("Clean average or modifiable mask not calculated")
            
            sum_variance_half_clean = np.zeros(self.shape, dtype=np.float64)
            sum_variance_clean = np.zeros(self.shape, dtype=np.float64)
            logger.info('Calculating clean variance ...')
            
            for i in tqdm(range(self.nframes), desc='Calculating clean variance'):
                img = self._get_img(i)
                processor = SingleProcessor(
                    img,
                    self.series_config,
                    self.series_result.mask_modifiable
                )
                single_result = processor.clean_img()
                
                if single_result.img_half_clean is not None and self.series_result.avg_half_clean is not None:
                    diff_half_clean = single_result.img_half_clean - self.series_result.avg_half_clean
                    sum_variance_half_clean += diff_half_clean ** 2
                
                if single_result.img_clean is not None and self.series_result.avg_clean is not None:
                    diff_clean = single_result.img_clean - self.series_result.avg_clean
                    sum_variance_clean += diff_clean ** 2
            
            self.series_result.var_half_clean = sum_variance_half_clean / self.nframes
            self.series_result.var_clean = sum_variance_clean / self.nframes
            logger.debug("Clean-variance calculated")
        except Exception as e:
            logger.error(f"Clean-variance calculation failed: {str(e)}")
            raise

    # =====================================================================
    # Public Methods
    # =====================================================================

    def process_series(self) -> SeriesResult:
        """Execute the complete series processing pipeline including averaging, masking and variance calculation."""
        try:
            logger.info("Starting series processing pipeline")

            # Step 1: Calculate direct average and binary average
            self._avg_direct()

            # Step 2: Create protection mask
            self._mask()

            # Step 3: Calculate clean average  
            self._avg_clean()

            # Step 4: Calculate variances
            self._var_direct()
            self._var_clean()

            logger.info("Series processing pipeline completed successfully")
            return deepcopy(self.series_result)
        except Exception as e:
            logger.error(f"Series processing pipeline failed: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up image series resources when the processor is deleted."""
        if hasattr(self, 'img_series'):
            self.img_series.cleanup()
            logger.debug("SeriesProcessor resources cleaned up")

    def __del__(self):
        """Automatically clean up resources when the processor is deleted."""
        self.cleanup()
