"""
Main controller for the GUI application.
"""
from typing import Dict, Any
from PySide6.QtCore import QObject, Slot

from ..models.processing_model import ProcessingModel
from ..views.main_view import MainView
from ...core import ProcessingParams

class MainController(QObject):
    """Controller that connects the model and view components."""
    
    def __init__(self, model: ProcessingModel, view: MainView):
        """
        Initialize the controller.
        
        Args:
            model: The processing model
            view: The main view
        """
        super().__init__()
        self.model = model
        self.view = view
        
        # Connect model signals to view
        self.model.status_changed.connect(self.view.log_message)
        self.model.progress_updated.connect(self.view.update_progress)
        self.model.error_occurred.connect(self.view.show_error)
        self.model.processing_finished.connect(self.on_processing_finished)
        
        # Connect view signals to controller slots
        self.view.loadDataButton.connect(self.on_load_data_clicked)
        self.view.loadParamsButton.connect(self.on_load_params_clicked)
        self.view.startButton.connect(self.on_start_clicked)
        self.view.saveDataButton.connect(self.on_save_data_clicked)
        self.view.saveParamsButton.connect(self.on_save_params_clicked)
        self.view.plotTypeComboBox.connect(self.on_plot_type_changed)
        self.view.plotButton.connect(self.on_plot_clicked)
        
        # Initialize view with default parameter values
        self.view.update_parameter_widgets(self.model.get_parameters())
                
    @Slot()
    def on_load_data_clicked(self):
        """Handle load data button click."""
        filename = self.view.get_open_filename("Select first image in series", "EDF files (*.edf)")
        if filename:
            if self.model.load_data(filename):
                self.view.show_info("Data loaded successfully!")
                
    @Slot()
    def on_load_params_clicked(self):
        """Handle load parameters button click."""
        filename = self.view.get_open_filename("Load Parameters", "JSON files (*.json)")
        if filename:
            if self.model.load_parameters(filename):
                self.view.update_parameter_widgets(self.model.get_parameters())
                self.view.show_info("Parameters loaded successfully!")
                
    @Slot()
    def on_save_params_clicked(self):
        """Handle save parameters button click."""
        # Read current parameter values from the view
        params = self.view.get_parameters_from_widgets()
        # Create a new ProcessingParams instance to validate them
        try:
            params_model = ProcessingParams(**params)
            filename = self.view.get_save_filename("Save Parameters", "JSON files (*.json)")
            if filename:
                if self.model.save_parameters(filename, params_model):
                    self.view.show_info("Parameters saved successfully!")
        except ValueError as e:
            self.view.show_error(str(e))
                
    @Slot()
    def on_start_clicked(self):
        """Handle start button click."""
        # Read current parameter values from the view
        params = self.view.get_parameters_from_widgets()
        # Create a new ProcessingParams instance to validate them
        try:
            params_model = ProcessingParams(**params)
            # Process the images with the validated parameters
            if self.model.process_images(params_model):
                # Processing is handled asynchronously via signals
                pass
        except ValueError as e:
            self.view.show_error(str(e))
            
    @Slot()
    def on_processing_finished(self):
        """Handle processing finished signal."""
        # Update the plot with the processed data
        plot_type = self.view.get_current_plot_type()
        img_data = self.model.get_plot_data(plot_type)
        if img_data is not None:
            self.view.plot_image(img_data)
            
    @Slot()
    def on_save_data_clicked(self):
        """Handle save data button click."""
        output_dir = self.view.get_directory("Select output directory")
        if output_dir:
            if self.model.save_results(output_dir):
                self.view.show_info("Results saved successfully!")
                
    @Slot(str)
    def on_plot_type_changed(self, plot_type: str):
        """
        Handle plot type selection change.
        
        Args:
            plot_type: New plot type
        """
        # Update model with the new plot type
        self.model.current_plot_type = plot_type
        
    @Slot()
    def on_plot_clicked(self):
        """Handle plot button click."""
        # Get the current plot type and retrieve the data
        plot_type = self.view.get_current_plot_type()
        img_data = self.model.get_plot_data(plot_type)
        if img_data is not None:
            self.view.plot_image(img_data) 