"""
Plot view for displaying images.
"""
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Using Qt5Agg backend since Qt6Agg is not available in matplotlib

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
    
from matplotlib.pyplot import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout

class PlotView(QWidget):
    """Widget for displaying images with matplotlib.
    
    This can be embedded in a QML UI by using a QWidget container.
    """
    
    def __init__(self):
        """Initialize the plot view."""
        super().__init__()
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def plot(self, img: np.ndarray):
        """
        Plot an image with hot colormap.
        
        Args:
            img: Image data to plot
        """
        self.ax.clear()
        self.ax.imshow(img, cmap='hot')
        self.canvas.draw() 