"""
Module for handling image series loading and management.

This module provides functionality for loading and managing series of XRD images
using different implementations (manual file loading or fabio.open_series).
"""
import fabio
from pathlib import Path
from abc import ABC, abstractmethod
import os
import numpy as np

# =====================================================================
# Base Image Series Class
# =====================================================================

class BaseImageSeries(ABC):
    """Base class for image series loading.
    
    This abstract base class defines the interface for different implementations
    of loading image series. Implementations must provide methods for:
    - Getting individual frames
    - Getting the number of frames
    - Cleaning up resources
    """
    
    @abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index.
        
        Args:
            index: Frame index to retrieve
            
        Returns:
            numpy.ndarray: The frame data as a numpy array
            
        Raises:
            IndexError: If the index is out of range
        """
        pass
    
    @property
    @abstractmethod
    def nframes(self) -> int:
        """Get the number of frames in the series.
        
        Returns:
            int: Total number of frames in the series
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources held by the implementation.
        
        This method should be called when the implementation is no longer needed
        to free up any resources (file handles, memory, etc.).
        """
        pass

# =====================================================================
# Image Series Implementations
# =====================================================================

class ManualImageSeries(BaseImageSeries):
    """Implementation for loading images manually from directory.
    
    This implementation loads images by scanning a directory for files matching
    a specific extension pattern. Files are loaded on demand when requested.
    
    Attributes:
        directory (Path): Directory containing the image files
        extension (str): File extension pattern to match
        files (List[Path]): List of sorted file paths
    """
    
    def __init__(self, directory: str, extension: str = "*.tif") -> None:
        """Initialize the image series implementation.
        
        Args:
            directory: Directory containing the image files
            extension: File extension pattern to match (default: "*.tif")
            
        Raises:
            ValueError: If no matching files are found
        """
        self.directory = Path(directory)
        self.extension = extension
        self._load_files()
        
    def _load_files(self) -> None:
        """Load and sort image files from the directory.
        
        Raises:
            ValueError: If no matching files are found
        """
        # Get all matching files and sort them
        self.files = sorted(self.directory.glob(self.extension))
        if not self.files:
            raise ValueError(f"No files matching extension '{self.extension}' found in {self.directory}")
        
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series.
        
        Returns:
            int: Total number of frames in the series
        """
        return len(self.files)
    
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index.
        
        Args:
            index: Frame index to retrieve
            
        Returns:
            numpy.ndarray: The frame data as a numpy array
            
        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < self.nframes:
            raise IndexError(f"Frame index {index} out of range [0, {self.nframes})")
        
        frame = fabio.open(str(self.files[index]))
        data = frame.data.copy()  # Make a copy since frame will be closed
        frame.close()
        return data

    def cleanup(self) -> None:
        """Clean up any resources held by the implementation.
        
        This implementation doesn't hold any resources that need cleanup.
        """
        pass

class FabioImageSeries(BaseImageSeries):
    """Implementation for loading images using fabio.open_series.
    
    This implementation uses fabio's built-in series loading functionality to
    handle image series. It's more efficient for certain file formats
    that support series loading.
    
    Attributes:
        img_series: Fabio image series object
    """
    
    def __init__(self, first_filename: str) -> None:
        """Initialize the image series implementation.
        
        Args:
            first_filename: Path to the first image in the series
            
        Raises:
            IOError: If the file cannot be opened
        """
        self.img_series = fabio.open_series(first_filename=first_filename)
    
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series.
        
        Returns:
            int: Total number of frames in the series
        """
        return self.img_series.nframes
    
    def get_frame(self, index: int) -> np.ndarray:
        """Get a frame at the specified index.
        
        Args:
            index: Frame index to retrieve
            
        Returns:
            numpy.ndarray: The frame data as a numpy array
            
        Raises:
            IndexError: If the index is out of range
        """
        frame = self.img_series.get_frame(index)
        return frame.data

    def cleanup(self) -> None:
        """Clean up any resources held by the implementation.
        
        Closes the fabio image series to free up resources.
        """
        if hasattr(self, 'img_series'):
            self.img_series.close()

# =====================================================================
# Main Image Series Class
# =====================================================================

class ImageSeries:
    """Factory class for creating image series implementations.
    
    This class provides a factory method to create appropriate image series
    implementations based on the input parameters. It supports both manual file
    loading and fabio's built-in series loading.
    """
    
    @classmethod
    def create(cls, first_filename: str, use_fabio: bool = False) -> BaseImageSeries:
        """Create a new image series implementation.
        
        Args:
            first_filename: Path to the first image or directory containing images
            use_fabio: Whether to use fabio.open_series instead of manual loading
            
        Returns:
            BaseImageSeries: An implementation of the image series interface
            
        Raises:
            ValueError: If no matching files are found (manual implementation)
            IOError: If the file cannot be opened (fabio implementation)
        """
        if use_fabio:
            return FabioImageSeries(first_filename)
        else:
            return ManualImageSeries(os.path.dirname(first_filename))
