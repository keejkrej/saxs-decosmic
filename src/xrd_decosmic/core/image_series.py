"""Module for handling image series loading and management using different implementations."""
import fabio
from pathlib import Path
from abc import ABC, abstractmethod
import os
import numpy as np
import tifffile

# =====================================================================
# Base Image Series Class
# =====================================================================

class BaseImageSeries(ABC):
    """Abstract base class defining the interface for image series loading implementations."""
    
    @abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index."""
        pass
    
    @property
    @abstractmethod
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources held by the implementation."""
        pass

# =====================================================================
# Image Series Implementations
# =====================================================================

class ManualImageSeries(BaseImageSeries):
    """Implementation for loading images manually from directory by scanning for matching files."""
    
    def __init__(self, directory: str, extension: str = "*.tif") -> None:
        """Initialize the image series implementation."""
        self.directory = Path(directory)
        self.extension = extension
        self._load_files()
        
    def _load_files(self) -> None:
        """Load and sort image files from the directory."""
        # Get all matching files and sort them
        self.files = sorted(self.directory.glob(self.extension))
        if not self.files:
            raise ValueError(f"No files matching extension '{self.extension}' found in {self.directory}")
        
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        return len(self.files)
    
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index."""
        if not 0 <= index < self.nframes:
            raise IndexError(f"Frame index {index} out of range [0, {self.nframes})")
        
        return tifffile.imread(str(self.files[index]))

    def cleanup(self) -> None:
        """Clean up any resources held by the implementation."""
        pass

class FabioImageSeries(BaseImageSeries):
    """Implementation for loading images using fabio's built-in series loading functionality."""
    
    def __init__(self, first_filename: str) -> None:
        """Initialize the image series implementation."""
        self.img_series = fabio.open_series(first_filename=first_filename)
    
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        return self.img_series.nframes
    
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index."""
        try:
            frame = self.img_series.get_frame(index)
            if hasattr(frame, 'data') and frame.data is not None:
                return frame.data
            else:
                raise IndexError(f"Frame {index} has no data")
        except Exception as e:
            raise IndexError(f"Failed to get frame {index}: {str(e)}")

    def cleanup(self) -> None:
        """Clean up any resources held by the implementation."""
        if hasattr(self, 'img_series'):
            self.img_series.close()

# =====================================================================
# Main Image Series Class
# =====================================================================

class ImageSeries:
    """Factory class for creating appropriate image series implementations."""
    
    @classmethod
    def create(cls, first_filename: str, use_fabio: bool = False) -> BaseImageSeries:
        """Create a new image series implementation."""
        if use_fabio:
            return FabioImageSeries(first_filename)
        else:
            return ManualImageSeries(os.path.dirname(first_filename))
