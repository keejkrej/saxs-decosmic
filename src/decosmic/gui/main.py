"""
Entry point for the GUI application using MVC pattern with PySide6 widgets.
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt

from .models.processing_model import ProcessingModel
from .views.main_view import MainView
from .controllers.main_controller import MainController

def main():
    """Start the GUI application."""
    # Set application attributes
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("XRD Decosmic")
    app.setOrganizationName("Decosmic Research")
    
    # Create MVC components
    model = ProcessingModel()
    view = MainView()
    controller = MainController(model, view)
    
    # Show the view
    view.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 