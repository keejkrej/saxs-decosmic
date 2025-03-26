"""
Processing model for the GUI application.
"""
import os
import json
from typing import Optional, Dict
import numpy as np
from PySide6.QtCore import QObject, Signal
from ...core.processing_params import ProcessingParams
from ...core.series_processor import SeriesProcessor

class ProcessingModel(QObject):
    """Model responsible for processing images and managing data state."""
    
    # Signals for UI updates
    status_changed = Signal(str)
    progress_updated = Signal(int)
    processing_finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self):
        """Initialize the processing model."""
        super().__init__()
        self.processor: Optional[SeriesProcessor] = None
        self.current_plot_type: str = 'Average'
    
    def load_data(self, filename: str) -> bool:
        """
        Load data from a file.
        
        Args:
            filename: Path to the first image in the series
            
        Returns:
            Whether the operation was successful
        """
        try:
            self.status_changed.emit(f"Loading {filename}...")
            self.processor = SeriesProcessor(filename)
            self.status_changed.emit(f"Loaded {self.processor.img_num} images")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load data: {str(e)}")
            return False
    
    def load_parameters(self, filename: str) -> bool:
        """
        Load parameters from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            Whether the operation was successful
        """
        try:
            self.status_changed.emit(f"Loading parameters from {filename}...")
            # Create params with defaults
            params = ProcessingParams()
            
            # Load and update from file
            with open(filename, 'r') as f:
                file_params = json.load(f)
                for key, value in file_params.items():
                    setattr(params, key, value)
            
            # Validate parameters before using them
            params.validate()
            
            if self.processor:
                self.processor.load_params(params)
                
            self.status_changed.emit("Parameters loaded successfully")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load parameters: {str(e)}")
            return False
    
    def save_parameters(self, filename: str, params_model: ProcessingParams) -> bool:
        """
        Save parameters to a JSON file.
        
        Args:
            filename: Path to save the JSON file
            params_model: ProcessingParams instance to save
            
        Returns:
            Whether the operation was successful
        """
        try:
            # Validate parameters before saving
            params_model.validate()
            
            self.status_changed.emit(f"Saving parameters to {filename}...")
            params_dict = {
                'th_donut': params_model.th_donut,
                'th_mask': params_model.th_mask,
                'th_streak': params_model.th_streak,
                'win_streak': params_model.win_streak,
                'exp_donut': params_model.exp_donut,
                'exp_streak': params_model.exp_streak
            }
            
            with open(filename, 'w') as f:
                json.dump(params_dict, f, indent=2)
                
            self.status_changed.emit("Parameters saved successfully")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to save parameters: {str(e)}")
            return False
    
    def process_images(self, params_model: ProcessingParams) -> bool:
        """
        Process the loaded images.
        
        Args:
            params_model: ProcessingParams instance with parameters
            
        Returns:
            Whether the operation was successful
        """
        if not self.processor:
            self.error_occurred.emit("No data loaded. Please load data first.")
            return False
            
        try:
            # Validate parameters before processing
            params_model.validate()
            
            # Additional range validation
            if params_model.win_streak <= 0:
                raise ValueError("Window size must be positive")
            if params_model.th_streak >= params_model.win_streak ** 2:
                raise ValueError("Streak threshold must be less than window size squared")
            
            self.status_changed.emit("Processing images...")
            self.processor.load_params(params_model)
            
            # Create progress callback
            def progress_callback(idx: int):
                progress_percent = int(idx / self.processor.img_num * 100)
                self.progress_updated.emit(progress_percent)
            
            # Process the images
            self.processor.avg_clean_img(progress_callback)
            
            self.progress_updated.emit(100)
            self.status_changed.emit("Processing completed")
            self.processing_finished.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Processing failed: {str(e)}")
            return False
    
    def save_results(self, output_dir: str) -> bool:
        """
        Save processing results to files.
        
        Args:
            output_dir: Directory to save the results
            
        Returns:
            Whether the operation was successful
        """
        if not self.processor:
            self.error_occurred.emit("No processed data. Please process data first.")
            return False
            
        try:
            self.status_changed.emit(f"Saving results to {output_dir}...")
            self.processor.save_results(output_dir)
            self.status_changed.emit("Results saved successfully")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to save results: {str(e)}")
            return False
    
    def get_plot_data(self, plot_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get the data for the requested plot type.
        
        Args:
            plot_type: Type of plot to retrieve data for, or None to use current type
            
        Returns:
            The image data or None if not available
        """
        if not self.processor:
            self.error_occurred.emit("No data available for plotting")
            return None
            
        if plot_type:
            self.current_plot_type = plot_type
            
        try:
            if self.current_plot_type == 'Average':
                return self.processor.img_avg
            elif self.current_plot_type == 'Clean':
                return self.processor.img_clean_avg
            elif self.current_plot_type == 'Difference':
                return self.processor.img_diff_avg
            elif self.current_plot_type == 'Mask':
                return self.processor.ring_mask.astype(np.int32)
            elif self.current_plot_type == 'Donut':
                return self.processor.sub_donut_avg
            elif self.current_plot_type == 'Streak':
                return self.processor.sub_streak_avg
            else:
                self.error_occurred.emit(f"Unknown plot type: {self.current_plot_type}")
                return None
        except Exception as e:
            self.error_occurred.emit(f"Failed to retrieve plot data: {str(e)}")
            return None
    
    def get_parameters(self) -> Dict[str, str]:
        """
        Get the default parameter values.
        
        Returns:
            Dictionary of parameter name -> value
        """
        # Create a ProcessingParams instance to get defaults
        params = ProcessingParams()
        return {
            'th_donut': params.th_donut,
            'th_mask': params.th_mask,
            'th_streak': params.th_streak,
            'win_streak': params.win_streak,
            'exp_donut': params.exp_donut,
            'exp_streak': params.exp_streak
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'processor') and self.processor:
            del self.processor 