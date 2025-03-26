"""
Main view for the XRD Decosmic application.
"""
from typing import Dict
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QComboBox,
    QProgressBar, QTextEdit, QGroupBox, QSpacerItem,
    QSizePolicy
)
from PySide6.QtCore import Signal

from .plot_view import PlotView
from .file_dialog import FileDialog

class MainView(QMainWindow):
    """Main view for the XRD Decosmic application."""
    
    # Signals for button clicks
    loadDataButton = Signal()
    loadParamsButton = Signal()
    startButton = Signal()
    saveDataButton = Signal()
    saveParamsButton = Signal()
    plotButton = Signal()
    plotTypeComboBox = Signal(str)
    
    def __init__(self):
        """Initialize the view."""
        super().__init__()
        
        # Create central widget
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        
        # Create file dialog helper
        self.file_dialog = FileDialog(self)
        
        # Create parameter widgets dictionary
        self.param_widgets = {}
        
        # Set up the UI
        self.setupUi()
        self.setupExtras()
        
    def setupUi(self):
        """Set up the UI with widgets."""
        # Main layout
        main_layout = QVBoxLayout(self.centralwidget)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        # Create and add buttons
        load_data_btn = QPushButton("Load Data")
        load_params_btn = QPushButton("Load Parameters")
        start_btn = QPushButton("Start")
        save_data_btn = QPushButton("Save Data")
        save_params_btn = QPushButton("Save Parameters")
        
        for btn in [load_data_btn, load_params_btn, start_btn, save_data_btn, save_params_btn]:
            btn.setFixedWidth(120)
            toolbar.addWidget(btn)
        
        toolbar.addStretch()
        main_layout.addLayout(toolbar)
        
        # Main content area
        content = QHBoxLayout()
        
        # Left panel - Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        
        # Donut Parameters
        param_layout.addWidget(QLabel("Donut Detection"))
        param_layout.addWidget(QLabel("Threshold:"))
        th_donut = QLineEdit()
        param_layout.addWidget(th_donut)
        self.param_widgets['th_donut'] = th_donut
        
        param_layout.addWidget(QLabel("Exponent:"))
        exp_donut = QLineEdit()
        param_layout.addWidget(exp_donut)
        self.param_widgets['exp_donut'] = exp_donut
        
        # Mask Parameter
        param_layout.addWidget(QLabel("Ring Mask"))
        param_layout.addWidget(QLabel("Threshold:"))
        th_mask = QLineEdit()
        param_layout.addWidget(th_mask)
        self.param_widgets['th_mask'] = th_mask
        
        # Streak Parameters
        param_layout.addWidget(QLabel("Streak Detection"))
        param_layout.addWidget(QLabel("Threshold:"))
        th_streak = QLineEdit()
        param_layout.addWidget(th_streak)
        self.param_widgets['th_streak'] = th_streak
        
        param_layout.addWidget(QLabel("Window Size:"))
        win_streak = QLineEdit()
        param_layout.addWidget(win_streak)
        self.param_widgets['win_streak'] = win_streak
        
        param_layout.addWidget(QLabel("Exponent:"))
        exp_streak = QLineEdit()
        param_layout.addWidget(exp_streak)
        self.param_widgets['exp_streak'] = exp_streak
        
        param_layout.addStretch()
        param_group.setFixedWidth(200)
        content.addWidget(param_group)
        
        # Right panel
        right_panel = QVBoxLayout()
        
        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Plot Type:"))
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Average", "Clean", "Difference", "Mask", "Donut", "Streak"])
        self.plot_type_combo.setFixedWidth(150)
        plot_controls.addWidget(self.plot_type_combo)
        
        plot_btn = QPushButton("Plot")
        plot_controls.addWidget(plot_btn)
        plot_controls.addStretch()
        right_panel.addLayout(plot_controls)
        
        # Plot area
        self.plot_view = PlotView()
        right_panel.addWidget(self.plot_view)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        right_panel.addWidget(self.progress_bar)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(100)
        right_panel.addWidget(self.log_text)
        
        content.addLayout(right_panel)
        main_layout.addLayout(content)
        
        # Connect signals
        load_data_btn.clicked.connect(self.loadDataButton.emit)
        load_params_btn.clicked.connect(self.loadParamsButton.emit)
        start_btn.clicked.connect(self.startButton.emit)
        save_data_btn.clicked.connect(self.saveDataButton.emit)
        save_params_btn.clicked.connect(self.saveParamsButton.emit)
        plot_btn.clicked.connect(self.plotButton.emit)
        self.plot_type_combo.currentTextChanged.connect(self.plotTypeComboBox.emit)
        
    def setupExtras(self):
        """Setup additional window properties."""
        self.setWindowTitle("XRD Decosmic")
        self.resize(1000, 700)
        
    def log_message(self, message: str):
        """
        Add a message to the log area.
        
        Args:
            message: Message to log
        """
        self.log_text.append(message)
        
    def update_progress(self, value: int):
        """
        Update the progress bar.
        
        Args:
            value: Percentage value (0-100)
        """
        self.progress_bar.setValue(value)
        
    def update_parameter_widgets(self, params: Dict[str, str]):
        """
        Update all parameter widgets with values from the model.
        
        Args:
            params: Dictionary of parameter name -> value
        """
        for name, value in params.items():
            if name in self.param_widgets:
                self.param_widgets[name].setText(value)
            
    def get_parameters_from_widgets(self) -> Dict[str, str]:
        """
        Get current parameter values from the widgets.
        
        Returns:
            Dictionary of parameter name -> value
        """
        return {name: widget.text() for name, widget in self.param_widgets.items()}
    
    def get_current_plot_type(self) -> str:
        """
        Get the currently selected plot type.
        
        Returns:
            Selected plot type
        """
        return self.plot_type_combo.currentText()
    
    def get_open_filename(self, title: str, filter_str: str) -> str:
        """
        Show file open dialog.
        
        Args:
            title: Dialog title
            filter_str: File filter string
            
        Returns:
            Selected filename or empty string if cancelled
        """
        return self.file_dialog.get_open_filename(title, filter_str)
    
    def get_save_filename(self, title: str, filter_str: str) -> str:
        """
        Show file save dialog.
        
        Args:
            title: Dialog title
            filter_str: File filter string
            
        Returns:
            Selected filename or empty string if cancelled
        """
        return self.file_dialog.get_save_filename(title, filter_str)
    
    def get_directory(self, title: str) -> str:
        """
        Show directory selection dialog.
        
        Args:
            title: Dialog title
            
        Returns:
            Selected directory or empty string if cancelled
        """
        return self.file_dialog.get_directory(title)
    
    def show_error(self, message: str):
        """
        Show error message.
        
        Args:
            message: Error message
        """
        self.file_dialog.show_error(message)
    
    def show_info(self, message: str):
        """
        Show info message.
        
        Args:
            message: Info message
        """
        self.file_dialog.show_info(message)
        
    def plot_image(self, img):
        """
        Plot an image in the plot area.
        
        Args:
            img: Image data to plot
        """
        if img is None:
            self.show_error("No image data available for plotting")
            return
        self.plot_view.plot(img) 