"""
Module for handling image series loading and management.
"""
import fabio
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
import os
import numpy as np

class ImageSeriesStrategy(ABC):
    """Strategy interface for image series loading."""
    
    @abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """
        Get a frame at the specified index.
        
        Args:
            index: Frame index
            
        Returns:
            numpy.ndarray: The frame data
        """
        pass
    
    @property
    @abstractmethod
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        pass

class ManualImageSeriesStrategy(ImageSeriesStrategy):
    """Strategy for loading images manually from directory."""
    
    def __init__(self, directory: str, extension: str = "*.tif"):
        """
        Initialize the image series strategy.
        
        Args:
            directory: Directory containing the image files
            extension: File extension to match (default: "*.tif")
        """
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
        """
        Get a frame at the specified index.
        
        Args:
            index: Frame index
            
        Returns:
            numpy.ndarray: The frame data
        """
        if not 0 <= index < self.nframes:
            raise IndexError(f"Frame index {index} out of range [0, {self.nframes})")
        
        frame = fabio.open(str(self.files[index]))
        try:
            return frame.data
        finally:
            frame.close()

class FabioImageSeriesStrategy(ImageSeriesStrategy):
    """Strategy for loading images using fabio.open_series."""
    
    def __init__(self, first_filename: str):
        """
        Initialize the image series strategy.
        
        Args:
            first_filename: Path to the first image in the series
        """
        self.img_series = fabio.open_series(first_filename=first_filename)
    
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        return self.img_series.nframes
    
    def get_frame(self, index: int) -> np.ndarray:
        """
        Get a frame at the specified index.
        
        Args:
            index: Frame index
            
        Returns:
            numpy.ndarray: The frame data
        """
        frame = self.img_series.get_frame(index)
        try:
            return frame.data
        finally:
            frame.close()

class ImageSeries:
    """Main class for image series management."""
    
    def __new__(cls, first_filename: str, use_fabio: bool = False) -> 'ImageSeries':
        """
        Create a new image series from a file.
        
        Args:
            first_filename: Path to the first image or directory containing images
            use_fabio: Whether to use fabio.open_series instead of manual loading
            
        Returns:
            An instance of ImageSeries
        """
        if use_fabio:
            strategy = FabioImageSeriesStrategy(first_filename)
        else:
            strategy = ManualImageSeriesStrategy(os.path.dirname(first_filename))
        
        instance = super().__new__(cls)
        instance._strategy = strategy
        return instance
    
    @property
    def nframes(self) -> int:
        """Get the number of frames in the series."""
        return self._strategy.nframes
    
    def get_frame(self, index: int) -> np.ndarray:
        """
        Get a frame at the specified index.
        
        Args:
            index: Frame index
            
        Returns:
            numpy.ndarray: The frame data
        """
        return self._strategy.get_frame(index)    
